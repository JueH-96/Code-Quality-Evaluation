def main():
    N, T, A = map(int, input().split())
    remaining = N - (T + A)
    
    # If Takahashi's current votes are so high that even if Aoki gets all remaining votes,
    # Aoki can't surpass Takahashi, then the outcome is decided in Takahashi's favor.
    if T > A + remaining:
        print("Yes")
        return
    
    # If Aoki's current votes are so high that even if Takahashi gets all remaining votes,
    # Takahashi can't surpass Aoki, then the outcome is decided in Aoki's favor.
    if A > T + remaining:
        print("Yes")
        return
    
    # Otherwise, the outcome is not decided yet.
    print("No")

# Do not forget to call main
if __name__ == "__main__":
    main()