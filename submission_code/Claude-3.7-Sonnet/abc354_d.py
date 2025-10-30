def main():
    A, B, C, D = map(int, input().split())
    
    # Calculate contributions to the final answer
    ans = (C - A) * (D - B)
    
    # Handle the even/odd cases based on the coordinates
    if (C+D) % 2 == (A+B) % 2:
        # If they have the same parity
        if (A+B) % 2 == 1:
            ans += 1
    else:
        # If they have different parity
        if (A+B) % 2 == 0:
            ans += 1
    
    print(ans)

if __name__ == "__main__":
    main()