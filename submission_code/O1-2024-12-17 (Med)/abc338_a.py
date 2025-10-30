def main():
    S = input().strip()
    if S[0].isupper() and all(ch.islower() for ch in S[1:]):
        print("Yes")
    else:
        print("No")

main()