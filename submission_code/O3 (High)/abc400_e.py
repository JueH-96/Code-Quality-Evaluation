import sys
import bisect

# ----------------------------------------------------------------------
#  Pre–compute every “400 number”   (definition in task statement)
# ----------------------------------------------------------------------
MAX_ROOT = 1_000_000              # because (10^6)^2 = 10^12

# smallest prime factor for every n ≤ MAX_ROOT
spf = [0] * (MAX_ROOT + 1)
for i in range(2, MAX_ROOT + 1):
    if spf[i] == 0:               # i is prime
        spf[i] = i
        if i * i <= MAX_ROOT:
            step = i
            for j in range(i * i, MAX_ROOT + 1, step):
                if spf[j] == 0:
                    spf[j] = i

# collect numbers whose square is a 400-number
four_hundred_numbers = []
append = four_hundred_numbers.append

for m in range(2, MAX_ROOT + 1):
    x = m
    distinct = 0
    while x > 1:
        p = spf[x]
        distinct += 1
        if distinct > 2:          # more than 2 prime factors – skip early
            break
        while x % p == 0:
            x //= p
    if distinct == 2:             # exactly two distinct primes
        append(m * m)             # its square fulfils the definition

# list is already sorted because m increased monotonically
# ----------------------------------------------------------------------
#  Answer queries
# ----------------------------------------------------------------------
def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    q, queries = data[0], data[1:]
    out_lines = []
    for a in queries:
        idx = bisect.bisect_right(four_hundred_numbers, a) - 1
        out_lines.append(str(four_hundred_numbers[idx]))
    sys.stdout.write('
'.join(out_lines))

if __name__ == '__main__':
    main()