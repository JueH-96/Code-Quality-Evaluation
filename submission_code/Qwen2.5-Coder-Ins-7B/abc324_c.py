import sys

def is_one_insertion_away(s1, s2):
    if len(s1) != len(s2) + 1:
        return False
    for i in range(len(s2) + 1):
        if s1[:i] + s1[i+1:] == s2:
            return True
    return False

def is_one_deletion_away(s1, s2):
    if len(s1) != len(s2) - 1:
        return False
    for i in range(len(s2)):
        if s1 == s2[:i] + s2[i+1:]:
            return True
    return False

def is_one_replacement_away(s1, s2):
    if len(s1) != len(s2):
        return False
    diff_count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            diff_count += 1
            if diff_count > 1:
                return False
    return diff_count == 1

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T_prime = data[1]
    S = data[2:]
    
    results = []
    for i, s in enumerate(S):
        if s == T_prime or is_one_insertion_away(s, T_prime) or is_one_deletion_away(s, T_prime) or is_one_replacement_away(s, T_prime):
            results.append(i + 1)
    
    print(len(results))
    print(' '.join(map(str, results)))

if __name__ == "__main__":
    main()