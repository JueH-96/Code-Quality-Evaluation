def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Fenwick (Binary Indexed) Tree helpers
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.data = [0]*(size+1)
        def update(self, i, val):
            # i is 1-based index in Fenwick usage
            while i <= self.size:
                self.data[i] += val
                i += i & -i
        def query(self, i):
            # Sum from 1..i
            s = 0
            while i > 0:
                s += self.data[i]
                i -= i & -i
            return s
        def range_query(self, l, r):
            return self.query(r) - self.query(l-1)

    # We will create two Fenwicks:
    # 1) fenwFreq: frequencies of encountered prefix-mod values
    # 2) fenwVal:  sum of indices (the actual prefix-mod value) for encountered prefix-mod values
    #
    # Because Fenwicks are typically 1-based, and our prefix mod values range is [0..M-1],
    # we will store them shifted by +1 for Fenwicks (i.e., value x is stored at index x+1).

    fenwFreq = FenwickTree(M)
    fenwVal  = FenwickTree(M)

    # pm[i] = prefix sums mod M.  pm[0] = 0
    # We keep track of the sum of all pm[i] so far to help with sum2 = sumSoFar - sum1
    sumSoFar_pm = 0  # sum of all pm[i] that have been added so far
    # Initially, pm[0] = 0 is present
    fenwFreq.update(1, 1)  # freq of 0 => index = 0+1
    fenwVal.update(1, 0)   # sum of values for '0'
    sumSoFar_pm += 0

    answer = 0
    prefix_mod = 0

    for i in range(N):
        prefix_mod = (prefix_mod + A[i]) % M
        x = prefix_mod

        # Count how many previous prefix_mods are <= x
        #   count1 = fenwFreq.query(x+1)
        #   sum1   = fenwVal.query(x+1)
        # Because x is in [0..M-1], we fenw-query up to x+1
        count1 = fenwFreq.query(x+1)
        sum1   = fenwVal.query(x+1)

        # total count of prefix_mod so far is (i+1), because we've added pm[0..i-1] previously
        # but let's just read from fenwFreq.query(M) if we want the total
        count_total = fenwFreq.query(M)
        sum_total   = fenwVal.query(M)

        count2 = (count_total - count1)
        sum2   = (sum_total - sum1)

        # sum for subarrays that end at i+1:
        #  For those prefix_mod <= x, (x - a)
        #  For those prefix_mod >  x, (x + M - a)
        # Translated:
        # part1 = count1*x - sum1
        # part2 = count2*(x + M) - sum2
        part1 = count1 * x - sum1
        part2 = count2 * (x + M) - sum2
        sum_for_this_r = part1 + part2

        answer += sum_for_this_r

        # Now add the current prefix_mod to Fenwicks
        fenwFreq.update(x+1, 1)  # freq of x
        fenwVal.update(x+1, x)   # sum of x
        sumSoFar_pm += x

    print(answer)