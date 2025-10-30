import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        R = int(data[index])
        B = int(data[index + 1])
        index += 2
        n = R + B
        
        if R == 0:
            if B % 2 != 0:
                out_lines.append("No")
            else:
                if B == 2:
                    out_lines.append("Yes")
                    out_lines.append("B 1 1")
                    out_lines.append("B 2 2")
                elif B == 4:
                    out_lines.append("Yes")
                    out_lines.append("B 2 1")
                    out_lines.append("B 3 2")
                    out_lines.append("B 2 3")
                    out_lines.append("B 1 2")
                else:
                    out_lines.append("No")
        else:
            if R % 2 != 0:
                out_lines.append("No")
            else:
                if R == 2 and B == 3:
                    out_lines.append("Yes")
                    out_lines.append("B 2 3")
                    out_lines.append("R 3 2")
                    out_lines.append("B 2 2")
                    out_lines.append("B 3 3")
                    out_lines.append("R 2 4")
                elif R == 4 and B == 0:
                    out_lines.append("Yes")
                    out_lines.append("R 1 1")
                    out_lines.append("R 1 2")
                    out_lines.append("R 2 2")
                    out_lines.append("R 2 1")
                else:
                    if B > 0:
                        start = (2, 1)
                        first_type = 'B'
                        count_blue = B - 1
                        count_red = R
                    else:
                        start = (1, 1)
                        first_type = 'R'
                        count_blue = 0
                        count_red = R - 1
                    
                    placed = set()
                    placed.add(start)
                    ans = [(first_type, start[0], start[1])]
                    current_type = first_type
                    current_pos = start
                    abort = False
                    
                    for i in range(n - 2):
                        candidates = []
                        r, c = current_pos
                        if current_type == 'B':
                            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                                nr = r + dr
                                nc = c + dc
                                if 1 <= nr <= 10**9 and 1 <= nc <= 10**9 and (nr, nc) not in placed:
                                    candidates.append((nr, nc))
                        else:
                            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                                nr = r + dr
                                nc = c + dc
                                if 1 <= nr <= 10**9 and 1 <= nc <= 10**9 and (nr, nc) not in placed:
                                    candidates.append((nr, nc))
                        
                        if not candidates:
                            abort = True
                            break
                        
                        next_type = None
                        next_pos = None
                        if count_red > 0 and count_blue > 0:
                            next_type = 'R'
                            next_pos = candidates[0]
                        elif count_red > 0:
                            next_type = 'R'
                            next_pos = candidates[0]
                        elif count_blue > 0:
                            next_type = 'B'
                            next_pos = candidates[0]
                        
                        placed.add(next_pos)
                        ans.append((next_type, next_pos[0], next_pos[1]))
                        if next_type == 'R':
                            count_red -= 1
                        else:
                            count_blue -= 1
                        current_type = next_type
                        current_pos = next_pos
                    
                    if abort or len(ans) < n - 1:
                        out_lines.append("No")
                    else:
                        r, c = current_pos
                        candidates = []
                        if current_type == 'B':
                            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                                nr = r + dr
                                nc = c + dc
                                if 1 <= nr <= 10**9 and 1 <= nc <= 10**9 and (nr, nc) not in placed:
                                    candidates.append((nr, nc))
                        else:
                            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                                nr = r + dr
                                nc = c + dc
                                if 1 <= nr <= 10**9 and 1 <= nc <= 10**9 and (nr, nc) not in placed:
                                    candidates.append((nr, nc))
                        
                        available_types = []
                        if count_red > 0:
                            available_types.append('R')
                        if count_blue > 0:
                            available_types.append('B')
                        
                        found_last = False
                        next_type = None
                        next_pos = None
                        for cand in candidates:
                            for tp in available_types:
                                if tp == 'B':
                                    if abs(cand[0] - start[0]) == 1 and abs(cand[1] - start[1]) == 1:
                                        next_type = tp
                                        next_pos = cand
                                        found_last = True
                                        break
                                else:
                                    if (abs(cand[0] - start[0]) == 1 and cand[1] == start[1]) or (abs(cand[1] - start[1]) == 1 and cand[0] == start[0]):
                                        next_type = tp
                                        next_pos = cand
                                        found_last = True
                                        break
                            if found_last:
                                break
                        
                        if not found_last:
                            out_lines.append("No")
                        else:
                            placed.add(next_pos)
                            ans.append((next_type, next_pos[0], next_pos[1]))
                            if next_type == 'R':
                                count_red -= 1
                            else:
                                count_blue -= 1
                            
                            if count_red != 0 or count_blue != 0:
                                out_lines.append("No")
                            else:
                                out_lines.append("Yes")
                                for piece in ans:
                                    out_lines.append(f"{piece[0]} {piece[1]} {piece[2]}")
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()