#include <iostream>
using namespace std;
int main()
{


float alexa_rank , score_final;
cin>>alexa_rank;

if (alexa_rank >= 1 && alexa_rank <= 1000)
	{
		score_final=((1001-alexa_rank)/1000)*10;
		score_final= score_final+90.46;
		cout<<score_final;
	}
if (alexa_rank >= 1001 && alexa_rank <= 9000)
	{
		score_final=((9001-alexa_rank)/9000)*10;
		score_final= score_final+86.32;
		cout<<score_final;
	}
if (alexa_rank >= 9001 && alexa_rank <= 15000)
	{
		score_final=((15001-alexa_rank)/15000)*10;
		score_final= score_final+79.83;
		cout<<score_final;
	}	
if (alexa_rank >= 15001 && alexa_rank <= 75000)
	{
		score_final=((75001-alexa_rank)/75000)*10;
		score_final= score_final+67.71;
		cout<<score_final;
	}
if (alexa_rank >= 75001 && alexa_rank <= 150000)
	{
		score_final=((150001-alexa_rank)/150000)*10;
		score_final= score_final+55.12;
		cout<<score_final;
	}
if (alexa_rank >= 150001 && alexa_rank <= 350000)
	{
		score_final=((350001-alexa_rank)/350000)*10;
		score_final= score_final+46.12;
		cout<<score_final;
	}
if (alexa_rank >= 350001 && alexa_rank <= 650000)
	{
		score_final=((650001-alexa_rank)/650000)*10;
		score_final= score_final+34.71;
		cout<<score_final;
	}
if (alexa_rank >= 650001 && alexa_rank <= 1250000)
	{
		score_final=((1250001-alexa_rank)/1250000)*10;
		score_final= score_final+28.7;
		cout<<score_final;
	}
if (alexa_rank >= 1250001 )
	{
		
		score_final=22.897;
		cout<<score_final;
	}
else 
 score_final=18.526; 						


return 0;	
}


			