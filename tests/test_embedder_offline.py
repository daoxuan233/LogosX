import os
import unittest

import numpy as np

from debate_arena.search.embedder import Embedder


class TestEmbedderOffline(unittest.TestCase):
    def test_hash_embedder_is_deterministic(self):
        os.environ["DEBATE_ARENA_EMBEDDER"] = "hash"
        os.environ["DEBATE_ARENA_EMBED_DIM"] = "64"
        e = Embedder()
        v1 = e.embed("你好，世界")
        v2 = e.embed("你好，世界")
        self.assertEqual(v1.shape, (64,))
        self.assertTrue(np.allclose(v1, v2))
        self.assertGreater(float(np.linalg.norm(v1)), 0.0)


if __name__ == "__main__":
    unittest.main()

