def main():
    L, R = map(int, input().split())
    
    if L == 1 and R == 0:
        print("Yes")       # Raising only left hand
    elif L == 0 and R == 1:
        print("No")        # Raising only right hand
    else:
        print("Invalid")   # Raising none or both

if __name__ == "__main__":
    main()