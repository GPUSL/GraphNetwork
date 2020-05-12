import matplotlib.pyplot as plt

# camrest BLEU
x=list(range(0,110,10))
# y1=[0.216,0.218,0.213,0.228,0.233,0.240,0.241,0.243,0.246,0.248,0.250]
# y2=[0.201,0.197,0.204,0.201,0.211,0.220,0.225,0.225,0.230,0.228,0.228]
# y3=[0.220,0.219,0.225,0.229,0.230,0.235,0.245,0.253,0.255,0.253,0.266]
# y4=[0.193,0.194,0.198,0.205,0.206,0.211,0.208,0.215,0.217,0.220,0.222]
#
# l1=plt.plot(x,y1,'r--',label='SEDST')
# l2=plt.plot(x,y2,'b--',label='SEDST/PR')
# l3=plt.plot(x,y3,'g--',label='SEDST+BERT')
# l4=plt.plot(x,y4,'y--',label='SEDST+BERT/PR')
# plt.plot(x,y1,'ro-',x,y2,'b+-',x,y3,'g^-',x,y4,'yo-')
# plt.title('BLEU Performance on Cambridge Restaurant corpus')
# plt.xlabel('Supervision Proportion / %')
# plt.ylabel('BLEU')
# plt.legend()
# plt.savefig('bleu_camrest.png')

#camrest joint acc
# y1=[0.765,0.779,0.829,0.882,0.937,0.957,0.955,0.957,0.966,0.960,0.973]
# y2=[0.730,0.755,0.781,0.871,0.883,0.920,0.913,0.922,0.928,0.930,0.929]
# y3=[0.770,0.792,0.830,0.893,0.949,0.960,0.959,0.968,0.974,0.973,0.978]
# y4=[0.761,0.772,0.773,0.783,0.791,0.801,0.813,0.828,0.840,0.844,0.844]
# l1=plt.plot(x,y1,'r--',label='SEDST')
# l2=plt.plot(x,y2,'b--',label='SEDST/PR')
# l3=plt.plot(x,y3,'g--',label='SEDST+BERT')
# l4=plt.plot(x,y4,'y--',label='SEDST+BERT/PR')
# plt.plot(x,y1,'ro-',x,y2,'b+-',x,y3,'g^-',x,y4,'yo-')
# plt.title('Joint Goal Accuracy Performance on Cambridge Restaurant corpus')
# plt.xlabel('Supervision Proportion / %')
# plt.ylabel('Joint Goal Accuracy')
# plt.legend()
# plt.savefig('acc_camrest.png')

#camrest Entity Match Rate
y1=[0.690,0.761,0.838,0.891,0.940,0.949,0.945,0.950,0.952,0.952,0.960]
y2=[0.422,0.734,0.800,0.862,0.931,0.918,0.920,0.921,0.925,0.926,0.926]
y3=[0.695,0.720,0.837,0.949,0.950,0.948,0.954,0.962,0.962,0.962,0.966]
y4=[0.759,0.772,0.775,0.788,0.812,0.820,0.822,0.845,0.842,0.840,0.842]
l1=plt.plot(x,y1,'r--',label='SEDST')
l2=plt.plot(x,y2,'b--',label='SEDST/PR')
l3=plt.plot(x,y3,'g--',label='SEDST+BERT')
l4=plt.plot(x,y4,'y--',label='SEDST+BERT/PR')
plt.plot(x,y1,'ro-',x,y2,'b+-',x,y3,'g^-',x,y4,'yo-')
plt.title('Entity Match Rate on Cambridge Restaurant corpus')
plt.xlabel('Supervision Proportion / %')
plt.ylabel('Entity Match Rate')
plt.legend()
plt.savefig('entity_camrest.png')