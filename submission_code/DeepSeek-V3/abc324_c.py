def is_possible(T_prime, S):
    len_T = len(T_prime)
    len_S = len(S)
    if len_S == len_T:
        # Check if they are the same or differ by one character
        diff = 0
        for i in range(len_T):
            if T_prime[i] != S[i]:
                diff += 1
                if diff > 1:
                    return False
        return True
    elif len_S == len_T + 1:
        # S is longer by one character, check if T_prime can be obtained by deleting one character from S
        i = j = 0
        diff = 0
        while i < len_T and j < len_S:
            if T_prime[i] == S[j]:
                i += 1
                j += 1
            else:
                j += 1
                diff += 1
                if diff > 1:
                    return False
        return True
    elif len_S == len_T - 1:
        # S is shorter by one character, check if T_prime can be obtained by inserting one character into S
        i = j = 0
        diff = 0
        while i < len_T and j < len_S:
            if T_prime[i] == S[j]:
                i += 1
                j += 1
            else:
                i += 1
                diff += 1
                if diff > 1:
                    return False
        return True
    else:
        return False

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    T_prime = data[1]
    S_list = data[2:2+N]
    result = []
    for idx, S in enumerate(S_list, 1):
        if is_possible(T_prime, S):
            result.append(idx)
    print(len(result))
    if result:
        print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()