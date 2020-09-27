#!/bin/bash

# CUR on Si data
sh run_cur.sh ACE Si 1 30 0
sh run_cur.sh SOAP Si 4 450 0
sh run_cur.sh SOAPLITE Si 4 450 0
sh run_cur.sh ACSF Si 1 51 0
sh run_cur.sh ACSFX Si 1 195 0
sh run_cur.sh CHSF Si 1 20 0
sh run_cur.sh MBTR Si 1 182 0

# CUR on TiO2 data
sh run_cur.sh ACE TiO2 1 30 0
sh run_cur.sh SOAP TiO2 17 1710 0
sh run_cur.sh SOAPLITE TiO2 14 1350 0
sh run_cur.sh ACSF TiO2 2 145 0
sh run_cur.sh ACSFX TiO2 5 534 0
sh run_cur.sh CHSF TiO2 1 40 0
sh run_cur.sh MBTR TiO2 9 900 0

# CUR on AlNiCu data
sh run_cur.sh ACE AlNiCu 1 30 0
sh run_cur.sh SOAP AlNiCu 38 3780 0
sh run_cur.sh SOAPLITE AlNiCu 27 2700 0
sh run_cur.sh ACSF AlNiCu 2 281 0
sh run_cur.sh ACSFX AlNiCu 5 544 0
sh run_cur.sh CHSF AlNiCu 1 40 0
sh run_cur.sh MBTR AlNiCu 24 2400 0

# CUR on CHON data
sh run_cur.sh ACE AlNiCu 1 30 0
sh run_cur.sh SOAP AlNiCu 66 6660 0
sh run_cur.sh SOAPLITE AlNiCu 45 4500 0
sh run_cur.sh ACSF AlNiCu 4 461 1
sh run_cur.sh ACSFX AlNiCu 10 1643 0
sh run_cur.sh CHSF AlNiCu 1 40 0
sh run_cur.sh MBTR AlNiCu 50 5000 0


