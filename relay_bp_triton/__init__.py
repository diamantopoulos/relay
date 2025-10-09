# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Relay-BP-S Triton GPU backend for quantum error correction decoding."""

from .decoder import RelayBPDecoder
from .graph import CSRGraph
from .utils import bitpack_errors, bitunpack_errors
from .adapter import RelayDecoder, ObservableDecoderRunner

__version__ = "0.1.0"
__all__ = ["RelayBPDecoder", "CSRGraph", "bitpack_errors", "bitunpack_errors", "RelayDecoder", "ObservableDecoderRunner"]
