def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    sticks_set = set()
    idx = 1
    
    for _ in range(N):
        s = data[idx]
        idx += 1
        r = s[::-1]
        # Use the lexicographically smaller between s and r as the canonical form
        sticks_set.add(min(s, r))
    
    print(len(sticks_set))

# Do not forget to call main()
if __name__ == "__main__":
    main()