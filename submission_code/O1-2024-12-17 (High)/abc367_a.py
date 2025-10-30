def main():
    A, B, C = map(int, input().split())
    if B < C:
        # Sleeping hours: [B, C)
        asleep = (B <= A < C)
    else:
        # Sleeping hours: [B, 24) and [0, C)
        asleep = (A >= B) or (A < C)
    print("No" if asleep else "Yes")

main()