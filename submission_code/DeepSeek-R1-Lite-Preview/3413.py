from typing import List

class Solution:
    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        n = len(skills)
        
        # Function to count consecutive wins before losing to a stronger player
        def count_wins_before_losing(queue, skills):
            n = len(queue)
            count = [0] * n
            stack = []
            for i in range(n - 1, -1, -1):
                while stack and skills[queue[i]] > skills[queue[stack[-1]]]:
                    count[i] += count[stack[-1]] + 1
                    stack.pop()
                stack.append(i)
            return count
        
        queue = list(range(n))
        current_streak = 0
        
        while True:
            # Find the position of the next player with higher skill than current_winner
            current_winner = queue[0]
            current_skill = skills[current_winner]
            next_stronger_pos = -1
            for i in range(1, len(queue)):
                if skills[queue[i]] > current_skill:
                    next_stronger_pos = i
                    break
            if next_stronger_pos == -1:
                # Current winner has the highest skill in the queue
                wins_per_cycle = len(queue) - 1
                if k <= current_streak + wins_per_cycle:
                    return current_winner
                else:
                    cycles_needed = (k - current_streak) // wins_per_cycle
                    remaining_wins = (k - current_streak) % wins_per_cycle
                    if remaining_wins == 0:
                        return current_winner
                    else:
                        current_streak += cycles_needed * wins_per_cycle
                        # Now, we need to reach remaining_wins
                        # Since the winner can cycle through the queue, we can directly return the winner
                        # because they can keep winning indefinitely against weaker players
                        return current_winner
            else:
                # Number of wins before losing
                wins_before_losing = next_stronger_pos - 1
                if current_streak + wins_before_losing >= k:
                    return current_winner
                else:
                    current_streak += wins_before_losing
                    # Remove the defeated players and set the next winner
                    queue = queue[next_stronger_pos:]
                    queue.append(queue.pop(0))  # Loser goes to the end