#!/bin/bash

# FPS on Si data
sh run_fps.sh ACE Si 1 30 0
sh run_fps.sh SOAP Si 4 450 0
sh run_fps.sh SOAPLITE Si 4 450 0
sh run_fps.sh ACSF Si 1 51 0
sh run_fps.sh ACSFX Si 1 195 0
sh run_fps.sh CHSF Si 1 20 0
sh run_fps.sh MBTR Si 1 182 0

# FPS on TiO2 data
sh run_fps.sh ACE TiO2 1 30 0
sh run_fps.sh SOAP TiO2 17 1710 0
sh run_fps.sh SOAPLITE TiO2 14 1350 0
sh run_fps.sh ACSF TiO2 2 145 0
sh run_fps.sh ACSFX TiO2 5 534 1
sh run_fps.sh CHSF TiO2 1 40 0
sh run_fps.sh MBTR TiO2 9 900 0

# FPS on AlNiCu data
sh run_fps.sh ACE AlNiCu 1 30 0
sh run_fps.sh SOAP AlNiCu 38 3780 0
sh run_fps.sh SOAPLITE AlNiCu 27 2700 0
sh run_fps.sh ACSF AlNiCu 2 281 0
sh run_fps.sh ACSFX AlNiCu 5 544 1
sh run_fps.sh CHSF AlNiCu 1 40 0
sh run_fps.sh MBTR AlNiCu 24 2400 0

# FPS on CHON data
sh run_fps.sh ACE AlNiCu 1 30 0
sh run_fps.sh SOAP AlNiCu 66 6660 0
sh run_fps.sh SOAPLITE AlNiCu 45 4500 0
sh run_fps.sh ACSF AlNiCu 4 461 0
sh run_fps.sh ACSFX AlNiCu 10 1643 1
sh run_fps.sh CHSF AlNiCu 1 40 0
sh run_fps.sh MBTR AlNiCu 50 5000 0

