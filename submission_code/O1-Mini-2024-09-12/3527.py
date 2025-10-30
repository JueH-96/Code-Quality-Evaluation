from typing import List
import bisect

class FenwickTree:
    def __init__(self, size):
        self.N = size + 2
        self.tree = [0] * self.N

    def update(self, index, delta):
        index +=1
        while index < self.N:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res =0
        index +=1
        while index >0:
            res += self.tree[index]
            index -= index & -index
        return res

    def query_range(self, left, right):
        return self.query(right) - self.query(left-1)

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        flags = [0]*n
        for i in range(n):
            flags[i] = 1 if colors[i] != colors[(i+1)%n] else 0

        runs = []
        i =0
        while i < n:
            if flags[i]==1:
                start =i
                while i <n and flags[i]==1:
                    i +=1
                runs.append( (start, i - start))
            else:
                i +=1
        # Handle circular merge
        if runs and runs[0][0]==0 and (runs[-1][0] + runs[-1][1] ==n):
            first = runs.pop(0)
            last = runs.pop(-1)
            runs.append( (last[0], last[1] + first[1]))
        
        # Initialize Fenwick Trees
        ft1 = FenwickTree(n)
        ft2 = FenwickTree(n)
        for run in runs:
            L = run[1]
            ft1.update(L, L +1)
            ft2.update(L,1)

        # Helper to add a run
        def add_run(L):
            ft1.update(L, L+1)
            ft2.update(L,1)
        
        # Helper to remove a run
        def remove_run(L):
            ft1.update(L, -(L+1))
            ft2.update(L, -1)

        # Runs sorted by start
        runs.sort()
        
        from collections import deque
        runs = deque(runs)

        # Function to find run index containing position
        def find_run(pos):
            l, r =0, len(runs)
            for idx, (start, length) in enumerate(runs):
                end = (start + length -1) %n
                if start <= end:
                    if start <= pos <= end:
                        return idx
                else:
                    if pos >= start or pos <= end:
                        return idx
            return -1

        answer = []
        for q in queries:
            if q[0]==1:
                s = q[1] -1
                if s <=0:
                    answer.append(0)
                    continue
                total1 = ft1.query(n)
                sum1 = ft1.query(n) - ft1.query(s-1)
                sum2 = ft2.query(n) - ft2.query(s-1)
                res = sum1 - s * sum2
                answer.append(res)
            else:
                _, j, c = q
                if colors[j]==c:
                    continue
                colors[j] =c
                # Update flags[j-1] and flags[j]
                affected = []
                for delta in [-1,0]:
                    idx = (j + delta) %n
                    old = flags[idx]
                    new =1 if colors[idx] != colors[(idx+1)%n] else 0
                    if old != new:
                        affected.append( (idx, old, new))
                        flags[idx]=new
                for idx, old, new in affected:
                    # Remove old run if old was 1
                    if old ==1:
                        run_idx = find_run(idx)
                        if run_idx ==-1:
                            continue
                        start, length = runs[run_idx]
                        remove_run(length)
                        runs.remove(runs[run_idx])
                        if length >1:
                            if idx == start:
                                new_start = start +1
                                new_length = length -1
                                runs.insert(run_idx, (new_start, new_length))
                                add_run(new_length)
                            elif idx == (start + length -1)%n:
                                new_length = length -1
                                runs.insert(run_idx, (start, new_length))
                                add_run(new_length)
                            else:
                                # Split into two runs
                                part1 = idx - start
                                part2 = length - part1 -1
                                runs.insert(run_idx, (start, part1))
                                add_run(part1)
                                runs.insert(run_idx +1, ((idx +1)%n, part2))
                                add_run(part2)
                    # Add new run if new is1
                    if new ==1:
                        left = (idx -1 +n)%n
                        right = (idx +1)%n
                        left_flag = flags[left]
                        right_flag = flags[right]
                        merge_left = left_flag ==1
                        merge_right = right_flag ==1
                        if merge_left and merge_right:
                            run_left = find_run(left)
                            run_right = find_run(right)
                            if run_left != -1 and run_right != -1 and run_left != run_right:
                                start1, len1 = runs[run_left]
                                start2, len2 = runs[run_right]
                                remove_run(len1)
                                remove_run(len2)
                                runs.remove(runs[run_right])
                                runs.remove(runs[run_left])
                                new_start = start1
                                new_len = len1 +1 + len2
                                runs.append( (new_start, new_len))
                                add_run(new_len)
                        elif merge_left:
                            run_left = find_run(left)
                            if run_left != -1:
                                start1, len1 = runs[run_left]
                                remove_run(len1)
                                runs.remove(runs[run_left])
                                new_start = start1
                                new_len = len1 +1
                                runs.append( (new_start, new_len))
                                add_run(new_len)
                        elif merge_right:
                            run_right = find_run(right)
                            if run_right != -1:
                                start2, len2 = runs[run_right]
                                remove_run(len2)
                                runs.remove(runs[run_right])
                                new_start = idx
                                new_len =1 + len2
                                runs.append( (new_start, new_len))
                                add_run(new_len)
                        else:
                            runs.append( (idx,1))
                            add_run(1)
        return answer