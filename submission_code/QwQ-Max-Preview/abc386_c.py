def can_insert(shorter, longer):
    i = j = 0
    count = 0
    while i < len(longer) and j < len(shorter):
        if longer[i] == shorter[j]:
            i += 1
            j += 1
        else:
            i += 1
            count += 1
            if count > 1:
                return False
    return j == len(shorter) and count <= 1

def main():
    import sys
    input = sys.stdin.read().split()
    K = int(input[0])
    S = input[1]
    T = input[2]
    
    if S == T:
        print("Yes")
        return
    
    len_s = len(S)
    len_t = len(T)
    
    if abs(len_s - len_t) > 1:
        print("No")
        return
    
    possible = False
    
    if len_s == len_t:
        diff = 0
        for a, b in zip(S, T):
            if a != b:
                diff += 1
                if diff > 1:
                    break
        possible = diff == 1
    elif len_s + 1 == len_t:
        possible = can_insert(S, T)
    elif len_s - 1 == len_t:
        possible = can_insert(T, S)
    
    print("Yes" if possible else "No")

if __name__ == "__main__":
    main()