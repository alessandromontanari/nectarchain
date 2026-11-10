import numpy as np
import pytest

from nectarchain.makers.extractor.charge_extractor import gradient_extractor


class TestGradientExtractor:
    """Unit tests for the gradient_extractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a gradient_extractor instance with default parameters."""
        from unittest.mock import MagicMock

        mock_subarray = MagicMock()
        return gradient_extractor(subarray=mock_subarray)

    @pytest.fixture
    def simple_pulse_waveform(self):
        """Create a simple pulse waveform (1 event, 1 pixel, N samples)."""
        n_samples = 50
        waveform = np.zeros((1, 1, n_samples), dtype=np.uint16)
        # Create a pulse: rise from 20-25, peak at 25, fall from 25-30
        rise_start, peak_pos, fall_end = 20, 25, 30
        waveform[0, 0, rise_start:peak_pos] = np.linspace(0, 100, peak_pos - rise_start)
        waveform[0, 0, peak_pos:fall_end] = np.linspace(100, 0, fall_end - peak_pos)

        return waveform

    @pytest.fixture
    def double_pulse_waveform(self):
        """Create a double pulse waveform (1 event, 1 pixel, N samples)."""
        n_samples = 50
        waveform = np.zeros((1, 1, n_samples), dtype=np.uint16)
        # Create two pulses: rise from 20-25,
        # fall from 25-30, rise 30-35, fall from 35-40
        rise_start, peak_pos1, valley_pos, peak_pos2, fall_end = 20, 25, 30, 35, 40
        waveform[0, 0, rise_start:peak_pos1] = np.linspace(
            0, 100, peak_pos1 - rise_start
        )
        waveform[0, 0, peak_pos1:valley_pos] = np.linspace(
            100, 80, valley_pos - peak_pos1
        )
        waveform[0, 0, valley_pos:peak_pos2] = np.linspace(
            80, 100, peak_pos2 - valley_pos
        )
        waveform[0, 0, peak_pos2:fall_end] = np.linspace(100, 0, fall_end - peak_pos2)

        return waveform

    @pytest.fixture
    def zero_pulse_waveform(self):
        """Create a zero pulse waveform (1 event, 1 pixel, N samples)."""
        n_samples = 50
        waveform = np.zeros((1, 1, n_samples), dtype=np.uint16)

        return waveform

    def test_import(self):
        """Test that the module can be imported."""
        from nectarchain.makers.extractor import charge_extractor

        assert hasattr(charge_extractor, "gradient_extractor")

    def test_default_parameters(self, extractor):
        """Test that default parameters are set correctly."""
        assert extractor.fixed_window == np.uint8(16)
        assert extractor.height_peak == np.uint8(10)

    def test_single_pulse_extraction(self, extractor, simple_pulse_waveform):
        """Test extraction on a simple pulse waveform."""
        peak_time, charge = extractor(
            simple_pulse_waveform, telid=1, selected_gain_channel=0
        )
        # Check that we got valid outputs
        assert peak_time is not None
        assert charge is not None
        # Check shapes (1 event, 1 pixel)
        assert peak_time.shape == (1, 1)
        assert charge.shape == (1, 1)
        # Check that we found a peak (values should be non-zero)
        assert peak_time[0, 0] > 0
        assert charge[0, 0] > 0

    def test_double_pulse_extraction(self, extractor, double_pulse_waveform):
        """Test extraction on a double pulse waveform."""
        peak_time, charge = extractor(
            double_pulse_waveform, telid=1, selected_gain_channel=0
        )
        # Check that we got valid outputs
        assert peak_time is not None
        assert charge is not None
        # Check shapes (1 event, 1 pixel)
        assert peak_time.shape == (1, 1)
        assert charge.shape == (1, 1)
        # Check that we found a peak (values should be non-zero)
        assert peak_time[0, 0] > 0
        assert charge[0, 0] > 0

    def test_zero_pulse_extraction(self, extractor, zero_pulse_waveform):
        """Test extraction on a zero pulse waveform."""
        peak_time, charge = extractor(
            zero_pulse_waveform, telid=1, selected_gain_channel=0
        )
        # Check that we got valid outputs
        assert peak_time is not None
        assert charge is not None
        # Check shapes (1 event, 1 pixel)
        assert peak_time.shape == (1, 1)
        assert charge.shape == (1, 1)
        # Check that we did not found a peak (values should be zero)
        assert np.array_equal(peak_time, np.zeros((1, 1)))
        assert np.array_equal(charge, np.zeros((1, 1)))

    def test_single_pulse_extraction_subped(self, extractor, simple_pulse_waveform):
        """Test extraction on a simple pulse waveform."""
        peak_time, charge = extractor(
            simple_pulse_waveform, telid=1, selected_gain_channel=0, subtract_ped=True
        )
        # Check that we got valid outputs
        assert peak_time is not None
        assert charge is not None
        # Check shapes (1 event, 1 pixel)
        assert peak_time.shape == (1, 1)
        assert charge.shape == (1, 1)
        # Check that we found a peak (values should be non-zero)
        assert peak_time[0, 0] > 0
        assert charge[0, 0] > 0
