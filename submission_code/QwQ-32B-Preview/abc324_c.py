import sys

def is_edit_distance_one(S, T):
    len_S = len(S)
    len_T = len(T)
    
    if len_S == len_T:
        diff_count = sum(1 for a, b in zip(S, T) if a != b)
        return diff_count <= 1
    elif len_S == len_T + 1:
        # Check if deleting one character from S makes it equal to T
        i = 0
        j = 0
        mismatched = False
        while i < len_S and j < len_T:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                if mismatched:
                    return False
                mismatched = True
                i += 1
        return j == len_T and i <= len_S
    elif len_S + 1 == len_T:
        # Check if inserting one character into S makes it equal to T
        i = 0
        j = 0
        mismatched = False
        while i < len_S and j < len_T:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                if mismatched:
                    return False
                mismatched = True
                j += 1
        return i == len_S and j <= len_T
    else:
        return False

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    T_prime = data[1]
    S = data[2:2+N]
    
    acceptable = []
    for idx, s in enumerate(S, start=1):
        if is_edit_distance_one(s, T_prime):
            acceptable.append(idx)
    
    acceptable.sort()
    print(len(acceptable))
    print(' '.join(map(str, acceptable)))

if __name__ == "__main__":
    main()