class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        # s = bindata, k = iterations, for pop count, pop the ones for s
        mod = int(1e9 + 7)  # 10^9 + 7
        
        # Get the binary integer stored by s
        b = int(s, 2)
        
        def findpop(data, iterations):
            dp, prev_d = [1], [1]
            for idx, c in enumerate(data):
                temp = []
                for bits, end in prev_d:
                    for i in range(2):
                        if i == 0:
                            EPOCH = (bits << 1) | ((c - '0') & i)  # Shift left by 1 space, and OR the shifted bits with current bits where c = '0'
                        else:
                            EPOCH = (bits + (c - '0')) & ((1 << (iterations)) - 1)  # Update the bits value by adding current bit c, then bitwise AND the bits with maximum possible bits count (containing all ones to 'iterations' length)
                        # If reached 0, add one else just add it to DP
                        temp.append((EPOCH, 0 if EPOCH == 0 else 1))
                prev_d = dp = [[sum(i) for i in x] for x in zip(*new)] for new in list(zip(*(sorted(temp), dp)))
            return sum(dp[1]) % mod

        return (findpop(s, k+1) - b) % mod