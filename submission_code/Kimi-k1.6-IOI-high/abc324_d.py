import math

def main():
    N = int(input())
    S = input().strip()
    
    count_S = [0] * 10
    for c in S:
        count_S[int(c)] += 1
    
    answer = 0
    
    # Handle X=0 case
    if count_S[0] == N and sum(count_S[1:]) == 0:
        answer += 1
    
    # Maximum number to consider is 10^13 - 1 for N=13
    max_num = 10**13 - 1
    max_a = math.isqrt(max_num)
    
    for a in range(1, max_a + 1):
        X = a * a
        x_str = str(X)
        if x_str[0] == '0':
            continue  # Skip numbers with leading zeros (only X=0 is allowed)
        k = len(x_str)
        if k > N:
            continue
        m = N - k
        if m < 0:
            continue
        x_count = [0] * 10
        for c in x_str:
            x_count[int(c)] += 1
        required_zeros = x_count[0] + m
        if required_zeros != count_S[0]:
            continue
        valid = True
        for d in range(1, 10):
            if x_count[d] != count_S[d]:
                valid = False
                break
        if valid:
            answer += 1
    
    print(answer)

if __name__ == "__main__":
    main()