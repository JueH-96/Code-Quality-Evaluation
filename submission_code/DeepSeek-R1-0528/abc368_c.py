def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    H = list(map(int, data[1:1+n]))
    
    total_moves = 0
    residue = 0
    for health in H:
        if residue == 0:
            if health == 0:
                x = 0
            else:
                mod5 = health % 5
                if mod5 == 0:
                    x = 3 * (health // 5)
                elif mod5 <= 2:
                    x = 3 * (health // 5) + mod5
                else:
                    x = 3 * (health // 5) + 3
        elif residue == 1:
            a = (health - 1) // 5
            rem = health - 5 * a
            if rem <= 1:
                x = 3 * a + 1
            elif rem <= 4:
                x = 3 * a + 2
            else:
                x = 3 * a + 3
        else:
            a = (health - 1) // 5
            rem = health - 5 * a
            if rem <= 3:
                x = 3 * a + 1
            elif rem == 4:
                x = 3 * a + 2
            else:
                x = 3 * a + 3
                
        total_moves += x
        residue = (residue + x) % 3
        
    print(total_moves)

if __name__ == "__main__":
    main()