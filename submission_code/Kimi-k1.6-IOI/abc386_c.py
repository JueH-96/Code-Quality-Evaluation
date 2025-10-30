import sys

def main():
    k = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()
    
    if s == t:
        print("Yes")
        return
    
    len_s = len(s)
    len_t = len(t)
    
    if abs(len_s - len_t) > 1:
        print("No")
        return
    elif len_s == len_t:
        diff = 0
        for a, b in zip(s, t):
            if a != b:
                diff += 1
                if diff > 1:
                    break
        print("Yes" if diff <= 1 else "No")
    else:
        longer, shorter = (s, t) if len_s > len_t else (t, s)
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
                    break
        count += (len(longer) - i)
        print("Yes" if count <= 1 else "No")

if __name__ == "__main__":
    main()