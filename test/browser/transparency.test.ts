/// <reference types="@webgpu/types" />

import { expect, test, vi } from 'vitest';
import { Input } from '../../src/input.js';
import { BufferSource, UrlSource } from '../../src/source.js';
import { ALL_FORMATS } from '../../src/input-format.js';
import { CanvasSink, ColorAlphaMerger, EncodedPacketSink, VideoSampleSink } from '../../src/media-sink.js';
import { Output } from '../../src/output.js';
import { WebMOutputFormat } from '../../src/output-format.js';
import { BufferTarget } from '../../src/target.js';
import { CanvasSource, VideoSampleSource } from '../../src/media-source.js';
import { canEncodeVideo, Quality } from '../../src/encode.js';
import { VideoSample } from '../../src/sample.js';
import { Conversion } from '../../src/conversion.js';
import { promiseWithResolvers } from '../../src/misc.js';

const createMergeFrames = (fullRange = true, timestamp = 123) => {
	const width = 10;
	const height = 4;
	const colorData = new Uint8Array(width * height * 4);
	const alphaData = new Uint8Array(width * height * 3 / 2).fill(128);
	const ramp = [0, 64, 128, 192, 255];
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			colorData.set(y < 2 ? [200, 100, 50, 255] : [50, 100, 200, 255], 4 * (y * width + x));
			const alpha = ramp[Math.floor(x / 2)]!;
			alphaData[y * width + x] = fullRange ? alpha : Math.round(16 + alpha * 219 / 255);
		}
	}

	const dimensions = { codedWidth: width, codedHeight: height, displayWidth: 20, displayHeight: 4, timestamp };
	const color = new VideoFrame(colorData, {
		...dimensions,
		duration: 456,
		format: 'RGBA',
		colorSpace: { primaries: 'bt709', transfer: 'iec61966-2-1', matrix: 'rgb', fullRange: true },
	});
	const alpha = new VideoFrame(alphaData, {
		...dimensions,
		format: 'I420',
		colorSpace: { primaries: 'bt709', transfer: 'iec61966-2-1', matrix: 'bt709', fullRange },
	});
	return { color, alpha };
};

const readMergedPixels = (frame: VideoFrame) => {
	const { width, height } = frame.visibleRect!;
	const context = new OffscreenCanvas(width, height).getContext('2d')!;
	context.drawImage(frame, 0, 0, width, height);
	return context.getImageData(0, 0, width, height).data;
};

test('GPU alpha merging preserves mask values and frame lifetime', async (context) => {
	const gpu = new ColorAlphaMerger();
	const cpu = new ColorAlphaMerger();
	const frames: VideoFrame[] = [];
	try {
		if (!(await gpu.initializeGpu())) {
			context.skip();
		}

		const { color, alpha } = createMergeFrames();
		frames.push(color, alpha);
		const expected = await cpu.merge(color.clone(), alpha.clone());
		frames.push(expected);
		const expectedPixels = readMergedPixels(expected);
		const workerSpy = vi.spyOn(Worker.prototype, 'postMessage');
		const copySpy = vi.spyOn(color, 'copyTo').mockRejectedValue(new Error('Unexpected color readback'));
		const alphaCopySpy = vi.spyOn(alpha, 'copyTo').mockRejectedValue(new Error('Unexpected alpha readback'));

		const next = createMergeFrames(false, 789);
		frames.push(next.color, next.alpha);
		const expectedNext = await cpu.merge(next.color.clone(), next.alpha.clone());
		frames.push(expectedNext);
		const expectedNextPixels = readMergedPixels(expectedNext);
		workerSpy.mockClear();
		const actualPromise = gpu.merge(color, alpha);
		const nextPromise = gpu.merge(next.color, next.alpha);
		const [actual, nextFrame] = await Promise.all([actualPromise, nextPromise]);
		frames.push(actual, nextFrame);
		gpu.close();

		expect(workerSpy).not.toHaveBeenCalled();
		expect(copySpy).not.toHaveBeenCalled();
		expect(alphaCopySpy).not.toHaveBeenCalled();
		expect(color.format).toBeNull();
		expect(alpha.format).toBeNull();
		expect(actual.timestamp).toBe(123);
		expect(actual.duration).toBe(456);
		expect(actual.displayWidth).toBe(20);
		expect(actual.displayHeight).toBe(4);
		expect(nextFrame.timestamp).toBe(789);
		const actualPixels = readMergedPixels(actual);
		const nextPixels = readMergedPixels(nextFrame);
		for (let i = 0; i < actualPixels.length; i++) {
			expect(Math.abs(actualPixels[i]! - expectedPixels[i]!)).toBeLessThanOrEqual(4);
			expect(Math.abs(nextPixels[i]! - expectedNextPixels[i]!)).toBeLessThanOrEqual(4);
		}
	} finally {
		frames.forEach(frame => frame.close());
		gpu.close();
		cpu.close();
	}
});

test('GPU merge failures fall back with live inputs and disable further GPU attempts', async (context) => {
	const merger = new ColorAlphaMerger();
	const frames: VideoFrame[] = [];
	try {
		if (!(await merger.initializeGpu())) {
			context.skip();
		}
		const importSpy = vi.spyOn(GPUDevice.prototype, 'importExternalTexture').mockImplementation(() => {
			throw new Error('Simulated import failure');
		});
		const workerSpy = vi.spyOn(Worker.prototype, 'postMessage');

		for (let i = 0; i < 2; i++) {
			const { color, alpha } = createMergeFrames();
			frames.push(color, alpha);
			const result = await merger.merge(color, alpha);
			frames.push(result);
			const pixels = readMergedPixels(result);
			expect(pixels[3]).toBe(0);
			expect(pixels[4 * 4 + 3]).toBe(128);
		}
		expect(importSpy).toHaveBeenCalledTimes(1);
		expect(workerSpy).toHaveBeenCalled();

		merger.close();
		const { color, alpha } = createMergeFrames();
		frames.push(color, alpha);
		await expect(merger.merge(color, alpha)).rejects.toThrow('merger closed');
		expect(color.format).toBeNull();
		expect(alpha.format).toBeNull();
	} finally {
		frames.forEach(frame => frame.close());
		merger.close();
	}
});

test('GPU merging leaves high-bit-depth frames on the existing CPU path', async (context) => {
	const merger = new ColorAlphaMerger();
	const frames: VideoFrame[] = [];
	try {
		if (!(await merger.initializeGpu())) {
			context.skip();
		}
		const importSpy = vi.spyOn(GPUDevice.prototype, 'importExternalTexture');
		const data = new Uint16Array([512, 512, 512, 512, 512, 512]);
		const init: VideoFrameBufferInit = {
			format: 'I420P10' as VideoPixelFormat,
			codedWidth: 2,
			codedHeight: 2,
			timestamp: 0,
			colorSpace: { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: true },
		};
		const color = new VideoFrame(data, init);
		const alpha = new VideoFrame(data, init);
		frames.push(color, alpha);
		const result = await merger.merge(color, alpha);
		frames.push(result);
		expect(importSpy).not.toHaveBeenCalled();
		expect(result.format).toBe('I420AP10');
	} finally {
		frames.forEach(frame => frame.close());
		merger.close();
	}
});

test.each([false, true])('Decoder preferences respect GPU availability and caller overrides (GPU=%s)', async (
	gpuAvailable,
) => {
	const initializeSpy = vi.spyOn(ColorAlphaMerger.prototype, 'initializeGpu').mockResolvedValue(gpuAvailable);
	vi.spyOn(ColorAlphaMerger.prototype, 'gpuEnabled', 'get').mockReturnValue(gpuAvailable);
	const configureSpy = vi.spyOn(VideoDecoder.prototype, 'configure').mockImplementation(() => {});
	vi.spyOn(VideoDecoder.prototype, 'decode').mockImplementation(() => {});
	const preferences = [undefined, 'no-preference', 'prefer-hardware', 'prefer-software'] as const;

	for (const fixture of ['transparency.webm', 'transparency-faulty.webm', 'demo.webm', 'demo.mp4']) {
		using input = new Input({ source: new UrlSource(`/${fixture}`), formats: ALL_FORMATS });
		const track = (await input.getPrimaryVideoTrack())!;
		const transparent = fixture.startsWith('transparency');
		for (const hardwareAcceleration of preferences) {
			initializeSpy.mockClear();
			configureSpy.mockClear();
			const decoder = await new VideoSampleSink(track, { hardwareAcceleration })._createDecoder(
				sample => sample.close(),
				(error) => { throw error; },
			);
			try {
				if (transparent) {
					const packet = (await new EncodedPacketSink(track).getFirstPacket())!;
					expect(packet.sideData.alpha).toBeDefined();
					decoder.decodeAlphaData(packet);
					expect(configureSpy.mock.lastCall![0].colorSpace?.transfer).toBe(
						gpuAvailable ? 'iec61966-2-1' : 'bt709',
					);
				}
				const automatic = hardwareAcceleration === undefined || hardwareAcceleration === 'no-preference';
				const expected = automatic && transparent && !gpuAvailable ? 'prefer-software' : hardwareAcceleration;
				expect(initializeSpy).toHaveBeenCalledTimes(transparent ? 1 : 0);
				expect(configureSpy).toHaveBeenCalledTimes(transparent ? 2 : 1);
				for (const [config] of configureSpy.mock.calls) {
					expect(config.hardwareAcceleration).toBe(expected);
				}
			} finally {
				decoder.close();
			}
		}
	}
});

test('GPU startup is bounded and closing prevents late initialization from reviving a merger', async (context) => {
	if (!navigator.gpu) {
		context.skip();
	}
	const adapter = promiseWithResolvers<GPUAdapter | null>();
	vi.spyOn(navigator.gpu, 'requestAdapter').mockReturnValue(adapter.promise);
	vi.useFakeTimers();
	const merger = new ColorAlphaMerger();
	const { color, alpha } = createMergeFrames();
	try {
		const ready = merger.initializeGpu();
		merger.close();
		await vi.advanceTimersByTimeAsync(1000);
		expect(await ready).toBe(false);
		adapter.resolve(null);
		await expect(merger.initializeGpu()).resolves.toBe(false);
		await expect(merger.merge(color, alpha)).rejects.toThrow('merger closed');
		expect(color.format).toBeNull();
		expect(alpha.format).toBeNull();
	} finally {
		adapter.resolve(null);
		color.close();
		alpha.close();
		merger.close();
		vi.useRealTimers();
	}
});

test('HDR and high-bit-depth VP9 do not opt into GPU merging', async () => {
	using input = new Input({ source: new UrlSource('/transparency.webm'), formats: ALL_FORMATS });
	const track = (await input.getPrimaryVideoTrack())!;
	const config = (await track.getDecoderConfig())!;
	vi.spyOn(track, 'canDecode').mockResolvedValue(true);
	const configSpy = vi.spyOn(track, 'getDecoderConfig');
	const hdrSpy = vi.spyOn(track, 'hasHighDynamicRange');
	const initializeSpy = vi.spyOn(ColorAlphaMerger.prototype, 'initializeGpu').mockResolvedValue(true);
	const configureSpy = vi.spyOn(VideoDecoder.prototype, 'configure').mockImplementation(() => {});

	for (const hdr of [false, true]) {
		configSpy.mockResolvedValue({ ...config, codec: hdr ? config.codec : 'vp09.02.10.10' });
		hdrSpy.mockResolvedValue(hdr);
		const decoder = await new VideoSampleSink(track)._createDecoder(
			sample => sample.close(),
			(error) => { throw error; },
		);
		try {
			expect(initializeSpy).not.toHaveBeenCalled();
			expect(configureSpy.mock.lastCall![0].hardwareAcceleration).toBe('prefer-software');
		} finally {
			decoder.close();
		}
	}
});

const decodeTransparentVideoTest = async () => {
	using input = new Input({
		source: new UrlSource('/transparency.webm'),
		formats: ALL_FORMATS,
	});

	const videoTrack = (await input.getPrimaryVideoTrack())!;
	expect(await videoTrack.canBeTransparent()).toBe(true);

	const sink = new VideoSampleSink(videoTrack);
	using sample = (await sink.getSample(0.5))!;

	expect(sample.format).toContain('A'); // Probably RGBA
	expect(sample.hasAlpha).toBe(true);

	const canvas = new OffscreenCanvas(sample.displayWidth, sample.displayHeight);
	const context = canvas.getContext('2d')!;

	sample.draw(context, 0, 0);

	const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
	expect(imageData.data[3]).toBeLessThan(255); // Check that there's actually transparent pixels
};

test('Can decode transparent video', async () => {
	await decodeTransparentVideoTest();
});

test('Can decode faulty transparent video and behaves gracefully', async () => {
	using input = new Input({
		source: new UrlSource('/transparency-faulty.webm'),
		formats: ALL_FORMATS,
	});

	const videoTrack = (await input.getPrimaryVideoTrack())!;
	const packetSink = new EncodedPacketSink(videoTrack);
	const secondKeyPacket = (await packetSink.getNextKeyPacket((await packetSink.getFirstPacket())!))!;

	const sink = new VideoSampleSink(videoTrack);

	using startSample = (await sink.getSample(await videoTrack.getFirstTimestamp()))!;
	expect(startSample.format).toContain('A');

	using secondSample = (await sink.getSample(secondKeyPacket.timestamp))!;
	expect(secondSample.format).not.toContain('A'); // There was no alpha key frame for this one
	expect(secondSample.hasAlpha).toBe(false);
});

test('Can extract transparent frames via CanvasSink', async () => {
	using input = new Input({
		source: new UrlSource('/transparency.webm'),
		formats: ALL_FORMATS,
	});

	const videoTrack = (await input.getPrimaryVideoTrack())!;
	const sink = new CanvasSink(videoTrack, { alpha: true });
	const wrappedCanvas = (await sink.getCanvas(await videoTrack.getFirstTimestamp()))!;

	const canvas = new OffscreenCanvas(wrappedCanvas.canvas.width, wrappedCanvas.canvas.height);
	const context = canvas.getContext('2d')!;
	context.drawImage(wrappedCanvas.canvas, 0, 0);

	let imageData = context.getImageData(0, 0, canvas.width, canvas.height);
	expect(imageData.data[3]).toBeLessThan(255); // Check that there's actually transparent pixels

	const opaqueSink = new CanvasSink(videoTrack); // Default is alpha: false
	const opaqueWrappedCanvas = (await opaqueSink.getCanvas(await videoTrack.getFirstTimestamp()))!;

	context.drawImage(opaqueWrappedCanvas.canvas, 0, 0);

	imageData = context.getImageData(0, 0, canvas.width, canvas.height);
	expect(imageData.data[3]).toBe(255);
});

const encodeTransparentVideoTest = async () => {
	const output = new Output({
		format: new WebMOutputFormat(),
		target: new BufferTarget(),
	});

	const canvas = new OffscreenCanvas(1280, 720);
	const context = canvas.getContext('2d')!;

	const source = new CanvasSource(canvas, {
		codec: 'vp9',
		quality: new Quality('high'),
		alpha: 'keep',
	});
	output.addVideoTrack(source);

	await output.start();

	context.fillStyle = '#ff0000';
	context.fillRect(200, 200, 200, 200);
	await source.add(0, 1);

	context.fillStyle = '#00ff00';
	context.fillRect(300, 300, 200, 200);
	await source.add(1, 1);

	context.fillStyle = '#0000ff';
	context.fillRect(400, 400, 200, 200);
	await source.add(2, 1);

	await output.finalize();

	const blob = new Blob([output.target.buffer!], {
		type: output.format.mimeType,
	});
	const url = URL.createObjectURL(blob);

	const video = document.createElement('video');
	video.src = url;
	video.muted = true;
	void video.play();

	await new Promise(resolve => video.addEventListener('loadeddata', resolve));

	// Let the video play for a little bit to prevent flake
	while (video.currentTime < 0.1) {
		await new Promise(resolve => setTimeout(resolve, 0));
	}

	expect(video.videoWidth).toBe(1280);
	expect(video.videoHeight).toBe(720);

	const probeCanvas = new OffscreenCanvas(1280, 720);
	const probeContext = probeCanvas.getContext('2d')!;

	probeContext.drawImage(video, 0, 0);

	let imageData = probeContext.getImageData(0, 0, probeCanvas.width, probeCanvas.height);
	const pos1 = { x: 100, y: 100 };
	const index1 = (pos1.x + pos1.y * probeCanvas.width) * 4;
	expect(imageData.data[index1 + 3]).toBe(0); // Transparent

	const pos2 = { x: 300, y: 300 }; // Dead center in the red square
	const index2 = (pos2.x + pos2.y * probeCanvas.width) * 4;

	// Red (within error)
	expect(imageData.data[index2 + 0]).greaterThanOrEqual(253);
	expect(imageData.data[index2 + 1]).lessThanOrEqual(3);
	expect(imageData.data[index2 + 2]).lessThanOrEqual(3);

	expect(imageData.data[index2 + 3]).toBe(255); // Opaque

	// Let's also check it's read correctly by Mediabunny
	using input = new Input({
		source: new BufferSource(output.target.buffer!),
		formats: ALL_FORMATS,
	});

	const videoTrack = (await input.getPrimaryVideoTrack())!;
	expect(await videoTrack.canBeTransparent()).toBe(true);

	const sink = new VideoSampleSink(videoTrack);

	using firstSample = (await sink.getSample(0))!;
	expect(firstSample.format).toContain('A');

	probeContext.clearRect(0, 0, probeCanvas.width, probeCanvas.height);
	firstSample.draw(probeContext, 0, 0);

	imageData = probeContext.getImageData(0, 0, probeCanvas.width, probeCanvas.height);

	expect(imageData.data[4 * (2 * probeCanvas.width + 2) + 3]).toBe(0); // Transparent
};

test('Can encode transparent video', async () => {
	await encodeTransparentVideoTest();
});

test('Can encode video with alternating transparency', async () => {
	// This test is already brutal when it comes to finding async race conditions, so run it thrice to be MORE brutal
	for (let j = 0; j < 3; j++) {
		const output = new Output({
			format: new WebMOutputFormat(),
			target: new BufferTarget(),
		});

		const canvas1 = new OffscreenCanvas(640, 480);
		const context1 = canvas1.getContext('2d', { alpha: true })!;
		context1.fillStyle = '#ff000080';
		context1.fillRect(0, 0, canvas1.width, canvas1.height);

		const canvas2 = new OffscreenCanvas(640, 480);
		const context2 = canvas2.getContext('2d', { alpha: false })!;
		context2.fillStyle = '#0000ff';
		context2.fillRect(0, 0, canvas2.width, canvas2.height);

		const source = new VideoSampleSource({
			codec: 'vp9',
			quality: new Quality('high'),
			alpha: 'keep',
		});
		output.addVideoTrack(source);

		await output.start();

		for (let i = 0; i < 64; i++) {
			using sample = new VideoSample(new Uint8Array(640 * 480 * 4), {
				format: i % 2 ? 'RGBX' : 'RGBA',
				codedWidth: 640,
				codedHeight: 480,
				timestamp: i,
				duration: 1,
			});
			await source.add(sample);
		}

		await output.finalize();

		using input = new Input({
			source: new BufferSource(output.target.buffer!),
			formats: ALL_FORMATS,
		});

		const videoTrack = (await input.getPrimaryVideoTrack())!;
		const packetSink = new EncodedPacketSink(videoTrack);

		let i = 0;
		for await (const packet of packetSink.packets()) {
			if (i % 2) {
				expect(packet.sideData.alpha).toBeUndefined();
			} else {
				expect(packet.sideData.alpha).toBeDefined();
			}

			i++;
		}

		const sampleSink = new VideoSampleSink(videoTrack);

		i = 0;
		for await (using sample of sampleSink.samples()) {
			if (i % 2) {
				expect(sample.format).not.toContain('A');
			} else {
				expect(sample.format).toContain('A');
			}

			i++;
		}
	}
});

test('Can encode transparent video with odd dimensions', async () => {
	const output = new Output({
		format: new WebMOutputFormat(),
		target: new BufferTarget(),
	});

	const canvas = new OffscreenCanvas(641, 479);
	const context = canvas.getContext('2d', { alpha: true })!;
	context.fillStyle = '#ff000080';
	context.fillRect(0, 0, canvas.width, canvas.height);

	const source = new CanvasSource(canvas, {
		codec: 'vp9',
		quality: new Quality('high'),
		alpha: 'keep',
	});
	output.addVideoTrack(source);

	await output.start();
	await source.add(0, 1);
	await output.finalize();
});

test('Positive encodability check with alpha', async () => {
	const result = await canEncodeVideo('vp9', { alpha: 'keep' });
	expect(result).toBe(true);
});

test('Can transmux transparent video, discards alpha by default', async () => {
	using input = new Input({
		source: new UrlSource('/transparency.webm'),
		formats: ALL_FORMATS,
	});
	const output = new Output({
		format: new WebMOutputFormat(),
		target: new BufferTarget(),
	});

	const conversion = await Conversion.init({
		input,
		output,
	});
	await conversion.execute();

	using outputInput = new Input({
		source: new BufferSource(output.target.buffer!),
		formats: ALL_FORMATS,
	});

	const videoTrack = (await outputInput.getPrimaryVideoTrack())!;
	expect(await videoTrack.canBeTransparent()).toBe(false);

	const sink = new VideoSampleSink(videoTrack);
	using sample = (await sink.getSample(await videoTrack.getFirstTimestamp()))!;
	expect(sample.hasAlpha).toBe(false);
});

test('Can transmux transparent video, can keep alpha', async () => {
	using input = new Input({
		source: new UrlSource('/transparency.webm'),
		formats: ALL_FORMATS,
	});
	const output = new Output({
		format: new WebMOutputFormat(),
		target: new BufferTarget(),
	});

	const conversion = await Conversion.init({
		input,
		output,
		video: {
			alpha: 'keep',
		},
	});
	await conversion.execute();

	using outputInput = new Input({
		source: new BufferSource(output.target.buffer!),
		formats: ALL_FORMATS,
	});

	const videoTrack = (await outputInput.getPrimaryVideoTrack())!;
	expect(await videoTrack.canBeTransparent()).toBe(true);

	const sink = new VideoSampleSink(videoTrack);
	using sample = (await sink.getSample(await videoTrack.getFirstTimestamp()))!;
	expect(sample.format).toContain('A');
	expect(sample.hasAlpha).toBe(true);
});

test('Can reencode transparent video, keeping alpha', async () => {
	using input = new Input({
		source: new UrlSource('/transparency.webm'),
		formats: ALL_FORMATS,
	});
	const output = new Output({
		format: new WebMOutputFormat(),
		target: new BufferTarget(),
	});

	const conversion = await Conversion.init({
		input,
		output,
		video: {
			width: 320,
			alpha: 'keep',
		},
		trim: {
			start: 0,
			end: 0.5,
		},
	});
	await conversion.execute();

	using outputInput = new Input({
		source: new BufferSource(output.target.buffer!),
		formats: ALL_FORMATS,
	});

	const videoTrack = (await outputInput.getPrimaryVideoTrack())!;
	expect(await videoTrack.canBeTransparent()).toBe(true);
	expect(await videoTrack.getDisplayWidth()).toBe(320);

	const sink = new VideoSampleSink(videoTrack);
	using sample = (await sink.getSample(await videoTrack.getFirstTimestamp()))!;
	expect(sample.format).toContain('A');
	expect(sample.hasAlpha).toBe(true);
});
