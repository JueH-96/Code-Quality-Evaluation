import sys

def can_form_t_from_s(S, T):
    len_s = len(S)
    len_t = len(T)

    for w in range(1, len_s):
        for c in range(1, w + 1):
            extracted = []
            for i in range(0, len_s, w):
                substring = S[i:i+w]
                if len(substring) >= c:
                    extracted.append(substring[c-1])
            if ''.join(extracted) == T:
                return "Yes"
    return "No"

def main():
    input = sys.stdin.read
    data = input().split()

    S = data[0]
    T = data[1]

    result = can_form_t_from_s(S, T)
    print(result)

if __name__ == "__main__":
    main()