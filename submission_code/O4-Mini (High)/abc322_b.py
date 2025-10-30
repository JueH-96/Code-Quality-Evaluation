def main():
    N, M = map(int, input().split())
    S = input().strip()
    T = input().strip()
    is_pref = T.startswith(S)
    is_suf = T.endswith(S)
    if is_pref and is_suf:
        print(0)
    elif is_pref:
        print(1)
    elif is_suf:
        print(2)
    else:
        print(3)

main()