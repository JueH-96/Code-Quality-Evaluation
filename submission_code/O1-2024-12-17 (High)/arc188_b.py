def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    answers = []
    for _ in range(t):
        n = int(data[idx]); k = int(data[idx+1])
        idx += 2
        
        # Special case:
        # If n == 2, there's only one possible k=1, and it is always "Yes".
        if n == 2:
            answers.append("Yes")
        # General case:
        # If n is even and k == n/2 (and n > 2), the two reflections coincide
        # in a way that is not the identity, leading to "No".
        elif n % 2 == 0 and 2 * k == n:
            answers.append("No")
        else:
            answers.append("Yes")
    
    print("
".join(answers))

# Do not forget to call main()!
if __name__ == "__main__":
    main()