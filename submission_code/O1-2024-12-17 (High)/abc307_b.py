def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    strings = data[1:]

    # Check all pairs of distinct i, j
    for i in range(N):
        for j in range(N):
            if i != j:
                combined = strings[i] + strings[j]
                if combined == combined[::-1]:  # Check palindrome
                    print("Yes")
                    return
    
    print("No")

# Do not forget to call main
if __name__ == "__main__":
    main()