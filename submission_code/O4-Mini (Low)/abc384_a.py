def main():
    import sys
    
    data = sys.stdin.read().split()
    N = int(data[0])
    c1 = data[1]
    c2 = data[2]
    S = data[3]
    
    # Build the result by replacing every character not equal to c1 with c2
    result = []
    for ch in S:
        if ch == c1:
            result.append(ch)
        else:
            result.append(c2)
    
    # Output the transformed string
    print(''.join(result))


if __name__ == "__main__":
    main()