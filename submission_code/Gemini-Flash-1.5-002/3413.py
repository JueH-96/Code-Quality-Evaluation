class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        queue = list(range(n))
        winner = -1
        win_count = 0
        while win_count < k:
            p1 = queue.pop(0)
            p2 = queue.pop(0)
            if skills[p1] > skills[p2]:
                queue.insert(0, p1)
                queue.append(p2)
                winner = p1
            else:
                queue.insert(0, p2)
                queue.append(p1)
                winner = p2
            win_count +=1

        return winner