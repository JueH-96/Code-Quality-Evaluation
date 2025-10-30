def main():
    # Read input
    n, c1, c2 = input().split()
    s = input().strip()
    
    # Build result by replacing every character not equal to c1 with c2
    result = []
    for ch in s:
        if ch == c1:
            result.append(ch)
        else:
            result.append(c2)
    
    # Output the result string
    print("".join(result))

# Call main to execute
if __name__ == "__main__":
    main()