import threading

import numpy as np
import sounddevice as sd
import ipywidgets as widgets


class Waveform:
    def __init__(
        self,
        *,
        freq: float = 440.0,
        amp: float = 0.5,
        phase_offset: float = 0.0,
        wave_type: str = "sine",
        enabled: bool = True,
    ):
        self.frequency = float(freq)
        self.amplitude = float(amp)
        self.phase_offset = float(phase_offset)
        self.wave_type = str(wave_type)
        self.enabled = bool(enabled)

        self.id = id(self)
        self.phase = 0.0

    def _wave_fn(self, phase_total: np.ndarray) -> np.ndarray:
        if self.wave_type == "sine":
            return np.sin(phase_total)
        if self.wave_type == "square":
            return np.sign(np.sin(phase_total))
        if self.wave_type == "sawtooth":
            return 2 * (phase_total / (2 * np.pi) - np.floor(phase_total / (2 * np.pi) + 0.5))
        if self.wave_type == "triangle":
            return 2 * np.abs(
                2 * (phase_total / (2 * np.pi) - np.floor(phase_total / (2 * np.pi) + 0.5))
            ) - 1
        return np.sin(phase_total)

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        if not self.enabled:
            return np.zeros(frames, dtype=np.float64)

        phase_increment = 2 * np.pi * self.frequency / sample_rate
        sample_indices = np.arange(frames)
        phase_array = self.phase + sample_indices * phase_increment
        phase_total = phase_array + self.phase_offset

        out = self.amplitude * self._wave_fn(phase_total)

        self.phase = (phase_array[-1] + phase_increment) % (2 * np.pi)
        return out


class RealTimeAudioPlayer:
    def __init__(self, *, sample_rate: int = 44100, buffer_size: int = 1024):
        self.sample_rate = int(sample_rate)
        self.buffer_size = int(buffer_size)

        self.is_playing = False
        self.stream: sd.OutputStream | None = None

        self.waveforms: list[Waveform] = []
        self.lock = threading.Lock()

    def add_waveform(
        self,
        *,
        freq: float = 440.0,
        amp: float = 0.5,
        phase_offset: float = 0.0,
        wave_type: str = "sine",
        enabled: bool = True,
    ) -> Waveform:
        with self.lock:
            wf = Waveform(
                freq=freq,
                amp=amp,
                phase_offset=phase_offset,
                wave_type=wave_type,
                enabled=enabled,
            )
            self.waveforms.append(wf)
            return wf

    def remove_waveform(self, waveform_id: int) -> None:
        with self.lock:
            self.waveforms = [w for w in self.waveforms if w.id != waveform_id]

    def audio_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            print(f"Stream status: {status}")

        with self.lock:
            wave = np.zeros(frames, dtype=np.float64)
            for wf in self.waveforms:
                wave += wf.render(frames, self.sample_rate)

        outdata[:, 0] = wave.astype(np.float32)

    def start(self) -> None:
        if self.is_playing:
            return

        self.is_playing = True
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.buffer_size,
            callback=self.audio_callback,
        )
        self.stream.start()
        print("Audio playback started")

    def stop(self) -> None:
        if not self.is_playing:
            return

        self.is_playing = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("Audio playback stopped")


class _WaveformControl:
    def __init__(self, waveform: Waveform, *, on_remove):
        self.waveform = waveform
        self._on_remove = on_remove

        self.wave_type_dropdown = widgets.Dropdown(
            options=["sine", "square", "sawtooth", "triangle"],
            value=waveform.wave_type,
            description="Wave",
        )

        self.enabled_checkbox = widgets.Checkbox(value=waveform.enabled, description="Enable")
        self.remove_button = widgets.Button(description="Remove")

        self.freq_slider = widgets.FloatSlider(
            value=waveform.frequency,
            min=20,
            max=2000,
            step=1,
            description=f"Freq {waveform.wave_type}",
            continuous_update=True,
        )

        self.amp_slider = widgets.FloatSlider(
            value=waveform.amplitude,
            min=0,
            max=1,
            step=0.01,
            description="Amp",
            continuous_update=True,
        )

        self.phase_slider = widgets.FloatSlider(
            value=waveform.phase_offset,
            min=0,
            max=2 * np.pi,
            step=0.01,
            description="Phase",
            continuous_update=True,
        )

        self.wave_type_dropdown.observe(self._on_wave_type_change, names="value")
        self.enabled_checkbox.observe(self._on_enabled_change, names="value")
        self.freq_slider.observe(self._on_freq_change, names="value")
        self.amp_slider.observe(self._on_amp_change, names="value")
        self.phase_slider.observe(self._on_phase_change, names="value")
        self.remove_button.on_click(self._on_remove_click)

        self.widget = widgets.VBox(
            [
                widgets.HBox([self.wave_type_dropdown, self.enabled_checkbox, self.remove_button]),
                self.freq_slider,
                self.amp_slider,
                self.phase_slider,
            ]
        )

    def _on_freq_change(self, change) -> None:
        self.waveform.frequency = float(change["new"])

    def _on_amp_change(self, change) -> None:
        self.waveform.amplitude = float(change["new"])

    def _on_phase_change(self, change) -> None:
        self.waveform.phase_offset = float(change["new"])

    def _on_wave_type_change(self, change) -> None:
        self.waveform.wave_type = str(change["new"])
        self.freq_slider.description = f"Freq {self.waveform.wave_type}"

    def _on_enabled_change(self, change) -> None:
        self.waveform.enabled = bool(change["new"])

    def _on_remove_click(self, _b) -> None:
        self._on_remove(self.waveform.id)


def create_synth_ui(*, fs: int = 44100, buffer_size: int = 1024):
    player = RealTimeAudioPlayer(sample_rate=fs, buffer_size=buffer_size)

    waveform_controls: list[_WaveformControl] = []

    controls_label = widgets.HTML("<h3>Waveform Controls</h3>")

    add_button = widgets.Button(description="Add Waveform")
    start_button = widgets.Button(description="Start Playback")
    stop_button = widgets.Button(description="Stop Playback")

    playback_controls = widgets.HBox([start_button, stop_button])

    main_ui = widgets.VBox([])

    def update_ui() -> None:
        if waveform_controls:
            waveform_widgets = widgets.VBox([c.widget for c in waveform_controls])
        else:
            waveform_widgets = widgets.HTML("No waveforms. Add one below!")

        main_ui.children = [controls_label, add_button, waveform_widgets, playback_controls]

    def remove_waveform(waveform_id: int) -> None:
        nonlocal waveform_controls
        player.remove_waveform(waveform_id)
        waveform_controls = [c for c in waveform_controls if c.waveform.id != waveform_id]
        update_ui()

    def add_new_waveform() -> None:
        wave_type = "sine"
        freq = 440 + len(waveform_controls) * 110

        wf = player.add_waveform(freq=freq, amp=0.3, phase_offset=0.0, wave_type=wave_type)
        control = _WaveformControl(wf, on_remove=remove_waveform)
        waveform_controls.append(control)
        update_ui()

    add_button.on_click(lambda _b: add_new_waveform())
    start_button.on_click(lambda _b: player.start())
    stop_button.on_click(lambda _b: player.stop())

    update_ui()
    return main_ui, player
