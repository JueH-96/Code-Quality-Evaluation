def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    
    # Check which persons are not eliminated
    candidates = [i for i in (1, 2, 3) if i != A and i != B]
    
    # If exactly one candidate remains, that's the culprit
    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()