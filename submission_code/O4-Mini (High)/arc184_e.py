def prefix_xor_int(x):
          shift = 1
          while shift < L:
              x ^= (x >> shift)
              shift <<= 1
          return x