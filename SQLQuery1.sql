create database BigData
use BigData;
create table info
(
	Id varchar(200), 
	AgencyName varchar(200),
	ComplaintType varchar(200),
	Borough varchar(200)
);

select * from info
insert into info values ('Test', 'Test', 'Test');

--1
select top(1) ComplaintType, count(*) as cnt from info
group by ComplaintType
order  by cnt desc

--2

select Borough, ComplaintType,  count(ComplaintType) as cnt 
from info as i
group by Borough, ComplaintType
having i.ComplaintType = (
					select top 1 i2.ComplaintType
					from info i2
					where i2.Borough = i.Borough
					group by i2.ComplaintType
					order by count(*) desc, i2.ComplaintType
				)			

--3
select top(1) AgencyName, count(*) as cnt from info
group by AgencyName
order  by cnt desc




delete from Info
