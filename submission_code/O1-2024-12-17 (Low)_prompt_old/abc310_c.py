def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    sticks = data[1:]
    
    distinct = set()
    for s in sticks:
        r = s[::-1]
        # Take the lexicographically smaller one as the canonical form
        canonical = min(s, r)
        distinct.add(canonical)
    
    print(len(distinct))

def main():
    solve()

if __name__ == "__main__":
    main()