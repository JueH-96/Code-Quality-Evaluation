import sys

def reconstruct_sequence(s):
    a = []
    i = 0
    while i < len(s):
        count = 0
        while s[i] == '-':
            count += 1
            i += 1
        a.append(count)
        i += 1
    return a

if __name__ == "__main__":
    s = input().strip()
    a = reconstruct_sequence(s)
    print(" ".join(map(str, a)))