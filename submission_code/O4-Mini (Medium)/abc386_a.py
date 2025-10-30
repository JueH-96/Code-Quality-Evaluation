def main():
    import sys
    from collections import Counter
    
    data = sys.stdin.read().strip().split()
    A, B, C, D = map(int, data)
    
    cnt = Counter((A, B, C, D))
    freqs = sorted(cnt.values())
    
    # A full house can be formed by adding one card iff
    # we currently have exactly two distinct values, and their counts are either [1,3] or [2,2].
    if len(freqs) == 2 and (freqs == [1, 3] or freqs == [2, 2]):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()