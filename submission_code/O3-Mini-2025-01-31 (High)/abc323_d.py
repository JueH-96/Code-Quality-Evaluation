import sys
import bisect
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    counts = defaultdict(int)
    # Read input: for each slime size S with count C, add to our dictionary.
    pos = 1
    for _ in range(N):
        s = int(data[pos]); c = int(data[pos+1])
        pos += 2
        counts[s] += c

    # We want to combine as many slimes as possible.
    # When we have two or more slimes of the same size X they can be merged
    # into one slime of size 2*X. Combining a pair reduces the total count by 1.
    # If we combine all possible pairs in every bucket, the remaining counts
    # becomes either 0 or 1. In addition, the combination produces slimes of a larger size,
    # and these may combine with existing slimes.
    #
    # So we simulate this process by processing the keys (slime sizes) in increasing order.
    # Because doubling always gives a larger number, new slimes are added later in the order.
    
    keys = sorted(counts.keys())
    # We maintain a set for quick membership check of keys in our sorted list.
    keys_set = set(keys)
    i = 0
    while i < len(keys):
        x = keys[i]
        cnt = counts[x]
        # Number of pairs we can combine.
        pairs = cnt // 2
        # Leave the remainder (0 or 1) in this bucket.
        counts[x] = cnt % 2
        if pairs:
            new_key = x * 2
            counts[new_key] += pairs
            # If new_key is new, add it to the sorted keys list.
            if new_key not in keys_set:
                bisect.insort(keys, new_key)
                keys_set.add(new_key)
        i += 1

    # In the final state every key has either 0 or 1 slime.
    # So the minimum number of slimes equals the sum of the remaining counts.
    answer = sum(counts[size] for size in keys)
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()