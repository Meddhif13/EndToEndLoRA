from unittest.mock import patch
import json

import sys
from pathlib import Path
from types import ModuleType

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide dummy modules for docling if not installed
if 'docling' not in sys.modules:
    mock_module = ModuleType('docling')
    dc = ModuleType('docling.document_converter')
    dc.DocumentConverter = object
    chunk = ModuleType('docling.chunking')
    chunk.HybridChunker = object
    sys.modules['docling'] = mock_module
    sys.modules['docling.document_converter'] = dc
    sys.modules['docling.chunking'] = chunk

from syntheticdatageneration import llm_call, Response


def fake_completion(*args, **kwargs):
    text = json.dumps(Response(generated=[]).dict())
    def gen():
        yield {"choices": [{"delta": {"content": text}}]}
    return gen()


def test_llm_call_parses_output():
    with patch('syntheticdatageneration.completion', fake_completion):
        result = llm_call('data')
    assert isinstance(result, dict)
    assert 'generated' in result
