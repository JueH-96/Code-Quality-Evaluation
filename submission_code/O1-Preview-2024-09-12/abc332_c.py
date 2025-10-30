# YOUR CODE HERE
def main():
    N, M = map(int, input().split())
    S = input().strip()
    N = int(N)
    M = int(M)
    S = S.strip()
    def is_possible(K):
        available_plain = M
        available_logo = K
        used_plain = 0
        used_logo = 0
        for c in S:
            if c == '1':
                if available_plain > 0:
                    available_plain -= 1
                    used_plain += 1
                elif available_logo > 0:
                    available_logo -= 1
                    used_logo +=1
                else:
                    return False
            elif c == '2':
                if available_logo > 0:
                    available_logo -=1
                    used_logo +=1
                else:
                    return False
            elif c == '0':
                # Wash all used T-shirts
                available_plain += used_plain
                used_plain = 0
                available_logo += used_logo
                used_logo = 0
            else:
                # Should not reach here
                return False
        return True
    # Binary search over K
    left = 0
    right = N
    while left < right:
        mid = (left + right) // 2
        if is_possible(mid):
            right = mid
        else:
            left = mid +1
    min_K = left
    print(min_K)

if __name__ == "__main__":
    main()