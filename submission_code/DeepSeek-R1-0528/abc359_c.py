def main():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())
    
    if sx == tx and sy == ty:
        print(0)
        return
        
    if (sx + sy) % 2 == 0:
        if sy == ty and (tx == sx or tx == sx + 1):
            print(0)
            return
    else:
        if sx == tx and (ty == sy or ty == sy + 1):
            print(0)
            return
            
    dx = abs(sx - tx)
    dy = abs(sy - ty)
    ans = max(dx, dy)
    print(ans)

if __name__ == '__main__':
    main()