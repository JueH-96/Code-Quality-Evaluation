def main():
    import sys
    import math

    input_data = sys.stdin.read().strip().split()
    T = int(input_data[0])
    idx = 1
    
    answers = []
    for _ in range(T):
        N = int(input_data[idx])
        K = int(input_data[idx + 1])
        idx += 2
        
        g = math.gcd(N, K)
        
        # If gcd(N, K) == 1, always "Yes"
        if g == 1:
            answers.append("Yes")
        # If gcd(N, K) == 2, we must check if K == N/2
        elif g == 2:
            if 2*K == N:
                # This corresponds to Bob being exactly opposite Alice
                answers.append("No")
            else:
                answers.append("Yes")
        else:
            # Otherwise "No"
            answers.append("No")
    
    print("
".join(answers))

# Do not forget to call main()!
if __name__ == "__main__":
    main()