import numpy as np
import PIL
import random
import matplotlib.pyplot as mplt
import prettyplotlib as pplt
import time

class Road:
	def __init__(self, roadLength, density, maxSpeed, beta=1, selfdriving=0, roadWidth=1, debug=False):
		self.len = roadLength
		self.width = roadWidth
		self.dens = density
		self.maxSpeed = maxSpeed
		self.maxSpeed1 = maxSpeed + 2	
		self.cars = int((self.len*self.dens)/self.width)
		self.speed = 0
		self.beta = beta
		self.debug = debug
		self.pers = selfdriving
		self.func2int = lambda x: int(x) if (x < self.maxSpeed + 1) else int(x-10)

		self._change_rate = 0.25

		if self.width == 1:
			self.roads = [0*x for x in range(self.len)]
			for i in range(self.cars):
				j = random.randint(0, self.len-1)
				if not self.roads[j]:
					if random.random() >= self.pers:
						self.roads[j] = float(random.randint(1, self.maxSpeed))
					else:
						self.roads[j] = float(random.randint(1, self.maxSpeed) + 10.0)

			temp = [x for x in self.roads if isinstance(x, float)]
			
			self._roads = [x for x in self.roads]
			self.speed = sum(temp)/len(temp)
		elif self.width == 2:
			self.road_1 = [0*x for x in range(self.len)]
			self.road_2 = [0*x for x in range(self.len)]
			for i in range(self.cars):
				j = random.randint(0, self.len-1)
				k = random.randint(0, self.len-1)
				if not self.road_1[j]:
					if random.random() >= self.pers:
						self.road_1[j] = float(random.randint(1, self.maxSpeed))
					else:
						self.road_1[j] = float(random.randint(1, self.maxSpeed) + 10.0)						
				if not self.road_2[k]:
					if random.random() >= self.pers:
						self.road_2[k] = float(random.randint(1, self.maxSpeed))
					else:
						self.road_2[k] = float(random.randint(1, self.maxSpeed) + 10.0)
			
			temp = [self.func2int(x) for x in self.road_1 if isinstance(x, float)] + [self.func2int(x) for x in self.road_2 if isinstance(x, float)] 
			
			self._road_1 = [x for x in self.road_1]
			self._road_2 = [x for x in self.road_2]
			self.speed = sum(temp)/len(temp)

	def changeSpeed(self, speed):
		self.speed = speed
	
	def stepForward(self, lamda=1):
		if self.debug:
			print 'road ready...'
			print map(self.func2int, self.road_1)
			print map(self.func2int, self.road_2)

			print '...'
		#########################################
		# 			Acceleration               	#
		#########################################
		'''           
		Single Lane        	  
		'''
		if self.width == 1:
			for i, cars in enumerate(self.roads):
				if isinstance(cars, float) and cars < self.maxSpeed:
					self.roads[i] = min(cars + 1, float(self.maxSpeed))
			#print 'acc\t:', self.roads
		'''
		Mutiple Lanes
		'''	
		if self.width == 2:
			#Flag = True

			for road in [self.road_1, self.road_2]:
				for i, car_v in enumerate(road):
					if isinstance(car_v, float):
						if car_v < self.maxSpeed:											# 一般的车
							road[i] = min(car_v + 1, float(self.maxSpeed))
						elif car_v > self.maxSpeed and self.func2int(car_v) < self.maxSpeed:# self-driving 的车	 
							for j in range(1, int(self.func2int(car_v))+1):
								try:
									if isinstance(road[i + j], float):						# 如果车前有车不加速
										road[i] = road[i + j]
										#Flag = False
										break
								except Exception, e:
									pass
							road[i] = min(car_v + 1, float(self.maxSpeed)+10)
							#if Flag:		                                                # 需要判断车距决定是不是加速
							#	road[i] = min(car_v + 1, float(self.maxSpeed)+10)

			if self.debug:
				print 'road acc...'
				print map(self.func2int, self.road_1)
				print map(self.func2int, self.road_2)

		#########################################
		# 			Deceleration               	#
		#########################################
		'''           
		Single Lane        	  
		'''
		if self.width == 1:
			for i, cars in enumerate(self.roads):
				if isinstance(cars, float):
					#print cars
					for j in range(1, int(cars)+1):
						#print i, ':\t', cars, '\t', 'j\t:', j
						try:
							if isinstance(self.roads[i + j], float):
								self.roads[i] = min(cars, (j - 1.0))
								break
							#print i, '>>>\t', self.roads[i]
						except Exception, e:
							pass
							#print Exception, ':', e
		'''
		Mutiple Lanes
		'''	
		if self.width == 2:
			for road in [self.road_1, self.road_2]:
				for i, car_v in enumerate(road):
					if isinstance(car_v, float):
						if car_v < self.maxSpeed + 1:										# 一般的车
							for j in range(1, int(car_v)+1):
								try:
									if isinstance(road[i + j], float):
										road[i] = min(car_v, j - 1.0)						#这里做了更改 j-1
										break
								except Exception, e:
									pass
						elif car_v > self.maxSpeed + 1: 								# self-driving 的车
							for j in range(1, int(self.func2int(car_v))+1):
								try:
									if isinstance(road[i + j], float):						
										#road[i] = min(car_v, (j + 9.0))					# 减速  j - 1 + 10,更改为 j + 10				
										road
										break
								except Exception, e:
									pass		                                                

			if self.debug:
				print 'road dec...'
				print map(self.func2int, self.road_1)
				print map(self.func2int, self.road_2)

		#########################################
		# 			Randomization               #
		#########################################
		'''           
		Single Lane        	  
		'''
		if self.width == 1:
			for i, cars in enumerate(self.roads):
				if isinstance(cars, float):
					if random.random() < lamda*(self.dens**(cars/float(self.maxSpeed))):
					#if random.random() < lamda:
						self.roads[i] = max(cars - 1.0, 1.0)
		'''
		Mutiple Lanes
		'''
		if self.width == 2:
			for road in [self.road_1, self.road_2]:
				for i, car_v in enumerate(road):
					if isinstance(car_v, float):
						if car_v < self.maxSpeed + 1:												# 一般的车
							if random.random() < lamda*(self.dens**(car_v/float(self.maxSpeed))):
								road[i] = max(car_v - 1.0, 0.0)
						elif car_v > self.maxSpeed:													# self-driving 的车 							
							pass
			if self.debug:
				print 'road random dec...'
				print map(self.func2int, self.road_1)
				print map(self.func2int, self.road_2)

		#########################################
		# 			 changeLane                 #
		#########################################
		if self.width == 2:
			self.road_1.reverse()
			self.road_2.reverse()
			Flag = False
			for i in range(self.len):
				if isinstance(self.road_1[i], float):
					if self.road_1[i] < self.maxSpeed + 1:
						for j in range(1, int(self.maxSpeed + 1)): 			# 判断是否前车距 gap 够 Vmax. 
							try:
								if isinstance(self.road_1[i - j], float): 
									Flag = True								# 车距不够就打算变道。
									break
							except:
								pass
						if Flag:
							for j in range(0, int(self.maxSpeed + 1)): 		# 判断旁边车道前车距 gap 够不够 Vmax.
								try:
									if isinstance(self.road_2[i - j], float):
										Flag = False 						# 旁车道前车距不够就不打算变道。
										break
								except:
									pass
							if Flag:
								for j in range(1, int(self.maxSpeed + 1)): 	# 判断旁边车道后车距 gap 够不够。
									try:
										if isinstance(self.road_2[i + j], float) and self.road_2[i + j] > j:
											Flag = False 					# 旁车道后距不够就不变道。
											break
									except:
										pass
								if Flag and random.random() <= self._change_rate:
									self.road_2[i] ,self.road_1[i] = self.road_1[i], self.road_2[i]
					else:													# self-driving 
						for j in range(1, int(self.maxSpeed + 1)): 			# 判断是否前车距 gap 够 Vmax. 
							try:
								if isinstance(self.road_1[i - j], float): 
									Flag = True								# 车距不够就打算变道。
									break
							except:
								pass
						if Flag:
							for j in range(0, int(self.maxSpeed + 1)): 		# 判断旁边车道前车距 gap 够不够 Vmax.
								try:
									if isinstance(self.road_2[i - j], float):
										Flag = False 						# 旁车道前车距不够就不打算变道。
										break
								except:
									pass
							if Flag:
								for j in range(1, int(self.maxSpeed + 1)): 	# 判断旁边车道后车距 gap 够不够。
									try:
										if isinstance(self.road_2[i + j], float) and self.road_2[i + j] > j:
											Flag = False 					# 旁车道后距不够就不变道。
											break
									except:
										pass
								if Flag:
									self.road_2[i] ,self.road_1[i] = self.road_1[i], self.road_2[i]
				
				elif isinstance(self.road_2[i], float):						# 另一条车道
					if self.road_2[i] < self.maxSpeed + 1:
						for j in range(1, int(self.maxSpeed + 1)): 			# 判断是否前车距 gap 够 Vmax. 
							try:
								if isinstance(self.road_2[i - j], float): 
									Flag = True								# 车距不够就打算变道。
									break
							except:
								pass
						if Flag:
							for j in range(0, int(self.maxSpeed + 1)): 		# 判断旁边车道前车距 gap 够不够 Vmax.
								try:
									if isinstance(self.road_1[i - j], float):
										Flag = False 						# 旁车道前车距不够就不打算变道。
										break
								except:
									pass
							if Flag:
								for j in range(1, int(self.maxSpeed + 1)): 	# 判断旁边车道后车距 gap 够不够。
									try:
										if isinstance(self.road_1[i + j], float) and self.road_1[i + j] > j:
											Flag = False 					# 旁车道后距不够就不变道。
											break
									except:
										pass
								if Flag and random.random() <= self._change_rate:
									self.road_2[i] ,self.road_1[i] = self.road_1[i], self.road_2[i]
					else:													# self-driving 
						for j in range(1, int(self.maxSpeed + 1)): 			# 判断是否前车距 gap 够 Vmax. 
							try:
								if isinstance(self.road_2[i - j], float): 
									Flag = True								# 车距不够就打算变道。
									break
							except:
								pass
						if Flag:
							for j in range(0, int(self.maxSpeed + 1)): 		# 判断旁边车道前车距 gap 够不够 Vmax.
								try:
									if isinstance(self.road_1[i - j], float):
										Flag = False 						# 旁车道前车距不够就不打算变道。
										break
								except:
									pass
							if Flag:
								for j in range(1, int(self.maxSpeed + 1)): 	# 判断旁边车道后车距 gap 够不够。
									try:
										if isinstance(self.road_1[i + j], float) and self.road_1[i + j] > j:
											Flag = False 					# 旁车道后距不够就不变道。
											break
									except:
										pass
								if Flag:
									#print '-------------------------------------------------------------------------------------------------------'
									self.road_2[i] ,self.road_1[i] = self.road_1[i], self.road_2[i]	
			self.road_1.reverse()
			self.road_2.reverse()
			if self.debug:
				print 'change...'
				print map(self.func2int, self.road_1)
				print map(self.func2int, self.road_2)

		#########################################
		# 			    forward                 #
		#########################################
		if self.width == 1:
			temp = [x for x in self.roads]
			#print 'dec\t:', temp
			self.roads = [0 for x in self.roads]

			for i, cars in enumerate(temp):
				if cars > 0 or isinstance(cars, float):
					if i + int(cars) > self.len-1:
						self.roads[i] = 0
					else:
						self.roads[i + int(cars)] = temp[i]
		
		if self.width == 2:

			road_1_temp = [x for x in self.road_1]
			road_2_temp = [x for x in self.road_2]

			self.road_1 = [0 for x in self.road_1]
			self.road_2 = [0 for x in self.road_2]

			for i, car_v in enumerate(road_1_temp):
				if isinstance(car_v, float):									# 第一条车道往前走
					if i + int(self.func2int(car_v)) > self.len - 1:
						self.road_1[i] = 0
					else:
						self.road_1[i + self.func2int(car_v)] = float(car_v)
				if road_2_temp[i] > 0 or isinstance(road_2_temp[i], float):		# 第二条车道往前走
					if i + int(self.func2int(road_2_temp[i])) > self.len - 1:
						self.road_2[i] = 0
					else:
						self.road_2[i + self.func2int(road_2_temp[i])] = float(road_2_temp[i])
			if self.debug:
				print 'forward...'			
				print map(self.func2int, self.road_1)
				print map(self.func2int, self.road_2)

		#########################################
		# 			Generation              	#
		#########################################
		if self.width == 1: 
			if not isinstance(self.roads[0], float):
				if random.random() < self.dens:
					self.roads[0] = float(random.randint(1, self.maxSpeed))
			
			speeds = [x for x in self.roads if isinstance(x, float)]
			
			self.changeSpeed(sum(speeds)/len(speeds))
			return map(int, self.roads)
		
		if self.width == 2:
			if not isinstance(self.road_1[0], float):
				if random.random() < self.beta*self.dens:
					if random.random() > self.pers:
						self.road_1[0] = float(random.randint(1, self.maxSpeed))
					else:
						self.road_1[0] = float(random.randint(1, self.maxSpeed) + 10.0)
			if not isinstance(self.road_2[0], float):
				if random.random() < self.beta*self.dens:
					if random.random() > self.pers:
						self.road_2[0] = float(random.randint(1, self.maxSpeed))			
					else:
						self.road_2[0] = float(random.randint(1, self.maxSpeed) + 10.0)			
			
			speeds = [self.func2int(x) for x in self.road_1 if isinstance(x, float)] + [self.func2int(x) for x in self.road_2 if isinstance(x, float)]
			
			self.changeSpeed(float(sum(speeds))/float(len(speeds)))
			if self.debug:
				print 'certain speed is :', self.speed
				print 'generation...'		
				print map(self.func2int, self.road_1)
				print map(self.func2int, self.road_2)
			return (map(int, self.road_1), map(int, self.road_2))

#########################################################################################

def plotA():
	color = ['c','m','y','k','r','g','b']
	for i, lamda in enumerate([0.05,0.06,0.08,0.10,0.15,0.30,0.50]):
		singleRoad = Road(600, 0.5, 5, 0.1)
		rr = np.array(map(int,singleRoad.roads))
		vv = [singleRoad.speed]
		for timestep in range(2500):
			#print singleRoad.roads
			r = singleRoad.stepForward(lamda)
			sp = singleRoad.speed
			vv.append(sp)
			#print r
			#print singleRoad.roads
			rr = np.row_stack((rr, r))
		rr = rr * 51
		#print rr.shape
		img = PIL.Image.fromarray(rr).convert('L')
		#img.show()
		img.save('.\img\Test\%03d-%s.jpg'%(i, lamda))
		singleRoad.roads = [x for x in singleRoad._roads]

		plt.plot(range(timestep + 2), vv, color[i],label= "%f"%lamda)
	plt.legend(loc='lower right', prop={'size':12})
	plt.xlabel('Time')
	plt.ylabel('Average Velocity')
	plt.savefig('01.png',dpi=300)
	#print rr

def plotB(lamda,enterProb,epoch):
	color = ['c','m','y','k','r','g','b']
	AverageV = []
	AverageFlow = []
	for i in range(epoch):
		for i, dens in enumerate([x*0.005 for x in range(1, 200)]):
			singleRoad = Road(600, dens, 5, 0.1, enterProb)
			vv = [singleRoad.speed]
			for timestep in range(2500):
				r = singleRoad.stepForward(lamda)
				sp = singleRoad.speed
				vv.append(sp)
			ck = vv[500:1000]
			cv = sum(ck)/len(ck)
			AverageV.append(cv)
			AverageFlow.append(cv*dens*3600)
			singleRoad.roads = [x for x in singleRoad._roads]
			#plt.plot(range(len(vv)), vv, color[i],label= "%f$"%dens)
		
		plt.scatter([x*0.005 for x in range(1, 200)], AverageFlow,c='m',s=25,alpha=0.4,marker='o')
		AverageFlow = []
		#plt.legend(loc='lower right', prop={'size':12})
	plt.xlabel('Average Density')
	plt.ylabel('Average Flow')
	plt.title('lambda%s-prob%s'%(lamda, enterProb))
	plt.savefig('sp-lamda%s-prob%s.png'%(lamda, enterProb), dpi=300)
	plt.figure()
	#print rr

def plotD_F(denslist, epoch=2, time=1500, roadLength=600, roadWidth=2, vmax=4, beta=1, sdrive=0, color='m',lim=(),debug=False, fl=True):
	AverageV = []
	AverageFlow = []
	if not fl:
		color = ['c','m','y','k','r','g','b','c','m','y','k','r','g','b','c','m','y','k','r','g','b']
	for i in range(epoch):  # simulation 的次数。
		#for i, dens in enumerate([x*0.1 for x in range(2, 9)]):
		for j, dens in enumerate(denslist):
			# __init__(self, roadLength, density, maxSpeed, beta=1, selfdriving=0, roadWidth=1, debug=False):
			twoLane = Road(roadLength, dens, vmax, beta=beta, selfdriving=sdrive, roadWidth=roadWidth, debug=debug)
			vv = [twoLane.speed]			
			for timestep in range(time):
				r = twoLane.stepForward()
				sp = twoLane.speed
				vv.append(sp)
			if fl:
				ck = vv[200:600]
				cv = float(sum(ck))/float(len(ck))
				AverageV.append(cv)
				AverageFlow.append(cv*dens*3600)
				twoLane.road_1 = [x for x in twoLane._road_1]            #?????需要吗？
				twoLane.road_2 = [x for x in twoLane._road_2]
			else:
				#pplt.plot(range(len(vv)), map(lambda x : float(2.0*7.5*x),vv), color[j],label= "%.2f"%dens)
				pplt.plot(range(len(vv)), map(lambda x : float(2.0*7.5*x),vv),label= "%.2f"%dens)
				pplt.legend(loc='lower right', prop={'size':10})
		
		if fl:	
			#plt.scatter(denslist, AverageFlow,c=color,s=25,alpha=0.4,marker='o')
			if i == 0: 
				pplt.scatter(denslist, AverageFlow,color=color,s=25,alpha=0.25,marker='o',label='%s%% Self-driving'%(int(sdrive*100)))
				AverageFlow = []
				pplt.legend(loc='upper left', prop={'size':9})
			else:
				pplt.scatter(denslist, AverageFlow,color=color,s=25,alpha=0.25,marker='o')
				AverageFlow = []
				pplt.legend(loc='lower right', prop={'size':9})	
	#mplt.ylim(lim)			
	mplt.xlabel('Time (/s)')
	mplt.ylabel('Average Velocity (miles/hour)')
	#mplt.savefig('Special_Dens_Flow_%s.png'%random.random(), dpi=300)
	mplt.savefig('Special_Time_AverageV_%s.png'%random.random(), dpi=300)

	

# __init__(self, roadLength, density, maxSpeed, beta=1, selfdriving=0, roadWidth=1, debug=False):
'''
magic = lambda x: int(x) if (x < 5) else int(x-10)

twoLane = Road(30, 0.8, 4,selfdriving=0, roadWidth=2,debug=True)
print map(magic, twoLane.road_1)
print map(magic, twoLane.road_2)

for i in range(10):
	#print map(int, twoLane.road_1)
	#print map(int, twoLane.road_2)
	r = twoLane.stepForward()
	#print map(int, twoLane.road_1)
	#print map(int, twoLane.road_2)
	print '=========================================================================================='
	print map(magic, twoLane.road_1)
	print map(magic, twoLane.road_2)
'''



# plotD_F(denslist, epoch=2, time=1500, roadLength=600, roadWidth=2, vmax=4, beta=1, sdrive=0, debug=False)
color1 = ['r','c','m','b']
color2 = ['c','m','b']
color3 = ['r','b']
#denslist = [x*0.005 for x in range(20, 200)]
#denslist = [x*0.005 for x in range(20,200)]
denslist = [x*0.1 for x in range(2,9)]
#print denslist
#denslist = [0.4]

#for i, sd in enumerate([0, 0.2, 0.5, 0.9]):
l1 = (0,5000)
l2 = (0,10000)
l3 = (0,10000)

plotD_F(denslist, epoch=1,time=60,roadLength=600,sdrive=1,color=color1[0],lim=l1,debug=False,fl=False)
#plotD_F(denslist, epoch=1,time=1000,roadLength=600,sdrive=0.9,color=color1[1],lim=l1,debug=False,fl=False)

'''
for i, sd in enumerate([0, 0.1]):
	start = time.clock()
	plotD_F(denslist, epoch=3,time=600,roadLength=600,sdrive=sd,color=color1[i],lim=l1,debug=False,fl=True)
	end = time.clock()

for i, sd in enumerate([0.1, 0.5, 0.9]):
	start = time.clock()
	plotD_F(denslist, epoch=3,time=600,roadLength=600,sdrive=sd,color=color2[i],lim=l2,debug=False,fl=True)
	end = time.clock()

for i, sd in enumerate([0, 1]):
	start = time.clock()
	plotD_F(denslist, epoch=3,time=600,roadLength=600,sdrive=sd,color=color3[i],lim=l3,debug=False,fl=True)
	end = time.clock()



for i, sd in enumerate([0.1, 0.5, 0.9]):
	start = time.clock()
	plotD_F(denslist, epoch=3,time=600,roadLength=600,sdrive=sd,color=color2[i],lim=l2,debug=False,fl=True)
	end = time.clock()
'''