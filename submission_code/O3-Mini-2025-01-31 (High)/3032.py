from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # Compute the number of levels needed (binary lifting levels)
        maxLevel = k.bit_length()  # 2^(maxLevel-1) <= k < 2^(maxLevel)

        # Build binary lifting tables:
        # jump[level][i] will be the node reached after 2^level passes starting from i.
        # sums[level][i] will be the sum of the nodes encountered in those 2^level passes (not including the starting i).
        jump = [receiver[:]]  # level 0: one pass, so jump[0][i] = receiver[i]
        sums = [receiver[:]]  # level 0: one pass, so the sum is receiver[i] itself

        # Precompute for all levels.
        for level in range(1, maxLevel):
            jmp_level = [0] * n
            sum_level = [0] * n
            prev_jump = jump[level - 1]
            prev_sum = sums[level - 1]
            for i in range(n):
                nxt = prev_jump[i]
                jmp_level[i] = prev_jump[nxt]
                sum_level[i] = prev_sum[i] + prev_sum[nxt]
            jump.append(jmp_level)
            sums.append(sum_level)

        # For every starting player x, compute f(x) using binary lifting.
        # f(x) = x + (sum over the ids encountered in exactly k passes)
        best = -10**18  # initialize with a very small number
        for x in range(n):
            total = x  # include starting player id
            cur = x
            # Process each bit of k; if bit i is set, use the precomputed 2^i jump.
            for bit in range(maxLevel):
                if k & (1 << bit):
                    total += sums[bit][cur]
                    cur = jump[bit][cur]
            if total > best:
                best = total
        return best

# The following code is provided to facilitate input/output operations.
def solve():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Process input.
    # We assume the input consists of two lines:
    # First line: the receiver array, e.g. "2 0 1"
    # Second line: the integer k, e.g. "4"
    # Alternatively, if the first token begins with '[', we adjust for that as well.
    try:
        if data[0].startswith('['):
            # Handle input like: receiver = [2,0,1]
            arr_str = ""
            idx = 0
            while idx < len(data):
                arr_str += data[idx] + " "
                if data[idx].endswith(']'):
                    break
                idx += 1
            arr_str = arr_str.strip().lstrip('[').rstrip(']')
            receiver = list(map(int, arr_str.replace(',', ' ').split()))
            if idx + 1 < len(data):
                k = int(data[idx + 1])
            else:
                k = int(sys.stdin.readline().strip())
        else:
            # Otherwise assume first line: receiver array, second token: k.
            # We assume all tokens except the last one belong to receiver.
            receiver = list(map(int, data[:-1]))
            k = int(data[-1])
    except Exception:
        return

    sol = Solution()
    ans = sol.getMaxFunctionValue(receiver, k)
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    solve()