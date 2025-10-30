def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    from math import gcd
    
    idx = 1
    answers = []
    for _ in range(t):
        N = int(input_data[idx]); idx+=1
        K = int(input_data[idx]); idx+=1
        g = gcd(N, K)
        if g == 1 or g == 2:
            answers.append("Yes")
        else:
            answers.append("No")
    
    print("
".join(answers))

# Don't forget to call main()
if __name__ == "__main__":
    main()