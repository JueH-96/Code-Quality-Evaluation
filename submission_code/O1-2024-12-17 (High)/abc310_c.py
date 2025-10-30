def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    idx = 1
    
    different_sticks = set()
    for _ in range(N):
        s = data[idx]
        idx += 1
        # Compute the canonical form by comparing the string with its reversal
        canonical = min(s, s[::-1])
        different_sticks.add(canonical)
    
    print(len(different_sticks))

# Do not forget to call main()!
if __name__ == "__main__":
    main()