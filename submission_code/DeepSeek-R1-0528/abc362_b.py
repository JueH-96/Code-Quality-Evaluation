def main():
    x1, y1 = map(int, input().split())
    x2, y2 = map(int, input().split())
    x3, y3 = map(int, input().split())
    
    dot1 = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1)
    dot2 = (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)
    dot3 = (x1 - x3) * (x2 - x3) + (y1 - y3) * (y2 - y3)
    
    if dot1 == 0 or dot2 == 0 or dot3 == 0:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()