class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        if len(nums) == 0 or len(nums) == 1:
            return []
        
        nums.sort()
        lst = []
        for i in range(0, len(nums)):
            
            start = i+1
            end = len(nums) -1
            
            if (i> 0  and nums[i] == nums[i-1]):
                continue
            
            while(start < end):
                
                if (end < len(nums) -1 and nums[end] == nums[end+1]):
                    end -=1
                    continue
                sum3  = nums[i] + nums[start] + nums[end]
                if ( sum3 == 0):
                    lst.append([nums[i], nums[start], nums[end]])
                    start +=1
                    end -=1
                elif sum3 < 0:
                    start +=1
                else:
                    end -=1
        return lst
      
      
      
      class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        
        
        ans = 0
        la = len(A)
        lb = len(B)
        lc = len(C)
        ld = len(D)
        m = {}
        
        for i in range(0, la):
            x = A[i]
            for j in range(0, lb):
                y = B[j]
                if x+y not in m:
                    m[x+y] = 0
                m[x+y] +=1
            
        for i in range(0, lc):
            x = C[i]
            for j in range(0, ld):
                y = D[j]
                
                target = -(x+y)
                
                if target in m:
                    
                    ans += m[target]
        return ans
      
      
      class Solution:
    def addBinary(self, a: str, b: str) -> str:
        lst = []
        i = len(a) - 1
        j = len(b) - 1
        carry = 0
        while(i >=0 or j>=0):
            
            current_sum = carry
            
            if i >=0:
                current_sum += int(a[i])
                i -=1
            if  j>=0:
                current_sum +=int(b[j])
                j-=1
                
            lst.insert(0, current_sum % 2)
            carry = current_sum // 2
        
        if carry > 0:
            lst.insert(0, carry)
        
        return ''.join(map(str, lst))
      
      class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        
        i = len(num1) - 1
        j = len(num2) - 1
        lst = []
        carry = 0
        sum1 = 0
        while( i >=0 or j >=0):
            sum1 = carry
            if  i >=0:
                sum1 += int(num1[i])
                i -=1
            if  j >=0:
                sum1 += int(num2[j])
                j -=1
                
            lst.insert(0, sum1 % 10)
            carry = sum1 // 10
            
        if carry !=0:
            lst.insert(0, 1)
            
        return ''.join(map(str, lst))
      
      
      
      # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        resultNode = ListNode(-1)
        head = resultNode
        p = l1
        q = l2
        sum1 = 0
        carry = 0
        while p is not None or q is not None:
            x = p.val if p is not None else 0
            y = q.val if q is not None else 0
            
            sum1 = carry + x + y
            
            head.next = ListNode(sum1 %10)
            carry = sum1//10
            head = head.next
            
            if p is not None:
                p = p.next
            if q is not None:
                q = q.next
                
        if carry > 0:
            head.next = ListNode(1)
            
        return resultNode.next
    
    
    # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        
        if not root:
            return True
        
        l , y = (self.binary_height_checker(root, is_balance = True))
        print(l)
        return y
            
            
    def binary_height_checker(self, root, is_balance):
        
        if root is None or not is_balance:
            return 0, is_balance
    
        lca, is_balance = self.binary_height_checker(root.left, is_balance)
        rca, is_balance = self.binary_height_checker(root.right, is_balance)
                
        if abs(lca- rca) > 1:
            is_balance = False
        
        return max(lca, rca) + 1, is_balance
            
        
        class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        if len(prices) == 1:
            return 0
            
        buy = prices[0]
        sell = prices[1]
        maxProfit = sell - buy
        
        for i in range(1, len(prices)):
            
            currentProfit = prices[i] - buy
            maxProfit = max(maxProfit, currentProfit)
            buy = min(buy, prices[i])
            
        return maxProfit if maxProfit > 0 else 0
            
        
        class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        
        if prices is None or len(prices) == 0:
            return 0
        profit = 0
        
        for i in range(0, len(prices)-1):
            if prices[i+1] > prices[i]:
                profit += (prices[i+1] - prices[i])
                
        return profit
    
    # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode, memo=[]) -> List[int]:
        lst = []
        self.helper(root, lst)
        return lst
        
    def helper(self, root, lst):
        if not root:
            return
        self.helper(root.left, lst)
        lst.append(root.val)
        self.helper(root.right, lst)
        
 # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        
        if root is None:
            return None
        
        queue = []
        result = []
        queue.append(root)
        
        while(queue):
            level = len(queue)
            lst = []
            
            for _ in range(0, len(queue)):
                node = queue.pop(0)
                lst.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(lst)
        return result
    
   # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        
        
        queue = deque([root])
        
        finalResult = []
        
        while queue:
            level = len(queue)
            currentNode = []
            
            for _ in range(level):
                
                node  = queue.popleft()
                
                if node:
                    currentNode.append(node.val)
                    queue.append(node.left)
                    queue.append(node.right)
            if currentNode:       
                finalResult = [currentNode] + finalResult
            
        return finalResult
            
  # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        lst = []
        self.helper(root, lst)
        return lst

    def helper(self, root, lst):
        if not root:
            return
        lst.append(root.val)
        self.helper(root.left, lst)
        self.helper(root.right, lst)
        
 # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        
        if not root:
            return None
        result = []
        queue = []
        queue.append(root)
        
        while (queue):
            level = len(queue)
            current = []
            
            for i in range(level):
                node  =  queue.pop(0)
                current.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(current[-1])
            
        return result
    
    # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        
        if root is None:
            return []
        
        zigzagVar = False
        finalResult = []
        queue = []
        queue.append(root)
        
        while(queue):    
            level = len(queue)
            currentList = []
            
            for  _ in range(level):    
                node = queue.pop(0)
                
                if zigzagVar:
                    currentList.insert(0, node.val)
                else:
                    currentList.append(node.val)
                    
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            zigzagVar = not zigzagVar
            finalResult.append(currentList)
            
        return finalResult
    
   class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        
        people.sort()
        num_boats = 0
        start = 0
        end = len(people) - 1
        
        while (start <=end):
            
            if start == end:
                num_boats +=1
                break
            if people[start] + people[end] <= limit:
                start +=1     
            end -=1
            num_boats +=1
        return num_boats
    
    class Solution:
    def climbStairs(self, n: int) -> int:
        
        l = self.fib(n,{})
        return l
        
    def fib(self, n, memo = {}):
        
        if n in memo:
            return memo[n]
        if n < 2:
            return 1
        
        memo[n] = self.fib(n-1) + self.fib(n-2)
        
        return memo[n]
        
    class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort()
        dp = [0]* (amount + 1)
        for i in range(len(dp)):
            dp[i] =  amount + 1
            
        dp[0] = 0
        
        for i in range(0, amount+1):
            for j in range(0, len(coins)):
                
                if coins[j] <= i:
                    dp[i] = min(dp[i], 1+ dp[i- coins[j]])
                else:
                    break
        if dp[amount] > amount:
            return -1
        else:
            return dp[amount]
     class Solution:
    def maxArea(self, height: List[int]) -> int:
        
         
        max_area = 0
        l = 0
        r = len(height)-1
    
        
        while(l < r):
            
            max_area = max(max_area, min(height[l], height[r])*(r-l))
            
            if height[l] < height[r]:
                l +=1
            else:
                r -=1
        return max_area
    
    class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        
        seen = defaultdict(int)
        
        for x in nums:
            if seen[x]:
                return True
            seen[x] +=1
        return False
    class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        dp = [0]*n      
        if s[0]!='0':
            dp[0] = 1
        for i in range(1, n):
            x = int(s[i])
            y = int(s[i-1:i+1])
            if x>=1 and x<=9:
                dp[i] += dp[i-1]
            if y >=10 and y <=26:
                if i-2 >=0:
                    dp[i] += dp[i-2]
                else:
                    dp[i] +=1           
        return dp[-1]
    
    class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        
        self.big = big
        self.medium = medium
        self.small = small
        
    def addCar(self, carType: int) -> bool:
        
        if carType == 1:
            
            if self.big > 0:
                self.big -=1
                return True
            else:
                return False
            
        elif carType == 2:
            if self.medium > 0:
                self.medium -=1
                return True
            else:
                return False
            
        elif carType == 3:
            if self.small > 0:
                self.small -=1
                return True
            else:
                return False
            

# Your ParkingSystem object will be instantiated and called as such:
# obj = ParkingSystem(big, medium, small)
# param_1 = obj.addCar(carType)

class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        
        res = []
        
        for i in range(len(nums)):
            
            if nums[abs(nums[i]) - 1] > 0:
                nums[abs(nums[i]) - 1] = - nums[abs(nums[i]) - 1]
            else:
                res.append(abs(nums[i]))
                
        return res
    
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        
        res = []
        
        for i in range(len(nums)):
            
            if nums[abs(nums[i]) - 1] > 0:    
                nums[abs(nums[i]) - 1] = - nums[abs(nums[i]) - 1]
                
            
        for i in range(len(nums)):
            
            if nums[i] > 0:
                res.append(i+1)
                
        return res
 class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:   
        l = 0
        r = len(nums)-1
        f_p = -1
        f_e = -1        
        if(len(nums) == 0):
            return[f_p,f_e]
            
        while(l <= r):
            
            if(nums[l] == target):
                f_p = l
                break
            l +=1
        while(r >=l):
             if(nums[r] == target):
                    f_e = r
                    break
            r -=1
        return [f_p, f_e]
                   
  import heapq
class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        
        self.left = []
        self.right = []
        

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        if not self.left or num <= -self.left[0]:
            heapq.heappush(self.left, -num)
            
        else:
            heapq.heappush(self.right, num)
            
        
        if abs(len(self.left) - len(self.right)) > 1:
            if(len(self.left) > len(self.right)):
               ele = -heapq.heappop(self.left)
               heapq.heappush(self.right, ele)
            else:
               temp = heapq.heappop(self.right)
               heapq.heappush(self.left, -temp)
               

    def findMedian(self):
        """
        :rtype: float
        """
               
        if len(self.left) == len(self.right):
               
               return (-self.left[0] + self.right[0] )/2.0
               
        elif len(self.left) > len(self.right):
               return -self.left[0]
        else:
               return self.right[0]
               
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        l =0
        r = len(nums)-1
        
        if nums[r] > nums[l]:
            return nums[l]
    
        
        while l <= r:
            
            mid = l+ (r-l) // 2
            
            
            if nums[mid] > nums[mid+1]:
                return nums[mid + 1]
            if nums[mid] < nums[mid - 1]:
                return nums[mid] 
            
            if nums[mid] > nums[l] and nums[mid] > nums[r]:
                
                l = mid + 1
                
            else:
                
                r = mid - 1
       
    
    
