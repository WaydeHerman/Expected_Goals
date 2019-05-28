"""
This file handles converting scraped JSON files to 'raw' tabular data.

usage:
    >>> define list of seasons ['1213', '1314', '1415', '1516', '1617', '1718', '1819']
    >>> define list of leagues ['EPL', 'LL', 'FL', 'ISA', 'GBL']
"""

import csv
import json
import sys
import ast

csv.field_size_limit(sys.maxsize)

INPUT_PATH = 'Data/JSON/'
OUTPUT_PATH = 'Data/raw/'

list_of_seasons = ['1213', '1314', '1415', '1516', '1617', '1718']
list_of_leagues = ['EPL', 'LL', 'FL', 'ISA', 'GBL']

def getJSON(league_var, season_var):
    """
    Function to extract features from JSON match events.

    :param league_var: League name (str).
    :param season_var: Season name (str).
    """
    list_of_JSON = []
    # Load the game events from the JSON csv:
    with open((INPUT_PATH + "JSON_{}{}.csv".format(league_var, season_var)), 'r') as jsonfile:
        fieldnames = ['league', 'season', 'homeTeam',
            'awayTeam', 'date', 'gameData']
        reader = csv.DictReader(jsonfile, fieldnames=fieldnames)
        for entries in reader:
            entryLeague = entries['league']
            entrySeason = entries['season']
            entryHomeTeam = entries['homeTeam']
            entryAwayTeam = entries['awayTeam']
            entryDate = entries['date']
            entryGameData = entries['gameData']
            entryDict = {'entryLeague': entryLeague, 'entrySeason': entrySeason, 'entryHomeTeam': entryHomeTeam, 'entryAwayTeam': entryAwayTeam,
                                        'entryDate': entryDate, 'entryGameData': entryGameData}
            list_of_JSON.append(entryDict)
        jsonfile.close()

    for jsons in list_of_JSON:
        gameData = ast.literal_eval(jsons['entryGameData'])
        league = jsons['entryLeague']
        season = jsons['entrySeason']
        homeTeam = jsons['entryHomeTeam']
        awayTeam = jsons['entryAwayTeam']
        date = jsons['entryDate']
        homeID = gameData["home"]["teamId"]
        awayID = gameData['away']['teamId']
        gameLength = gameData['minuteExpanded']
        hState = 0
        aState = 0

        # Determines the GK's ID for each team (for dribbles on the GK):
        gk_home_id = []
        gk_away_id = []
        shotDatabase = []
        for gkevents in gameData['events']:
            for gkeventsb in gkevents['qualifiers']:
                if 'GoalKick' in gkeventsb['type']['displayName']:
                    if gkevents['teamId'] == homeID:
                        if gkevents['playerId'] not in gk_home_id:
                            gk_home_id.append(int(gkevents['playerId']))
                    if gkevents['teamId'] == awayID:
                        if gkevents['playerId'] not in gk_away_id:
                            gk_away_id.append(int(gkevents['playerId']))

        iD = -1
        for event in gameData['events']:
            iD += 1
            if 'isShot' in event:
                if event['teamId'] == homeID:
                    shotTeam = homeTeam
                    shotState = hState
                elif event['teamId'] == awayID:
                    shotTeam = awayTeam
                    shotState = aState
                shotPlayerID = event['playerId']
                shotTeamID = event['teamId']
                shotMin = event['expandedMinute']
                shotSec = event['second']
                shotX = event['x']
                shotY = event['y']
                shotAssistID = None
                shotGoalYN = 0
                shotHeaderYN = 0
                shotBigChanceYN = 0
                shotFromCornerYN = 0
                shotFastBreakYN = 0
                shotPenaltyYN = 0
                shotDirectFKYN = 0
                shotOwnGoalYN = 0
                shotCrossYN = 0
                shotThroughballYN = 0
                shotIndirectFKYN = 0
                shotOnTarget = 0
                shot6Yard = 0
                shotPenaltyArea = 0
                shotOutBox = 0
                shotChanceCreatorID = None
                shotChanceX1 = None
                shotChanceY1 = None
                shotChanceX2 = None
                shotChanceY2 = None
                if event['type']['displayName'] == 'Goal':
                    shotGoalYN = 1
                    if shotTeam == homeTeam:
                        hState += 1
                        aState -= 1
                    elif shotTeam == awayTeam:
                        hState -= 1
                        aState += 1
                for qualifiers in event['qualifiers']:
                    if 'Head' in qualifiers['type']['displayName']:
                        shotHeaderYN = 1
                    if 'BigChance' in qualifiers['type']['displayName']:
                        shotBigChanceYN = 1
                    if 'FromCorner' in qualifiers['type']['displayName']:
                        shotFromCornerYN = 1
                    if 'FastBreak' in qualifiers['type']['displayName']:
                        shotFastBreakYN = 1
                    if 'Penalty' in qualifiers['type']['displayName']:
                        shotPenaltyYN = 1
                    if 'DirectFreekick' in qualifiers['type']['displayName']:
                        shotDirectFKYN = 1
                    if 'OwnGoal' in qualifiers['type']['displayName']:
                        shotOwnGoalYN = 1
                    if 'IntentionalAssist' in qualifiers['type']['displayName']:
                        IntentionalAssistYN = 1
                for satEvents in event['satisfiedEventsTypes']:
                    if satEvents == 8:
                        shotOnTarget = 1
                    if satEvents == 0:
                        shot6Yard = 1
                    if satEvents == 1:
                        shotPenaltyArea = 1
                    if satEvents == 2:
                        shotOutBox = 1
                # Want own goals to be measured but no detail taken:
                if shotOwnGoalYN != 1:
                    if 'relatedEventId' in event:
                        for eventassist in gameData['events']:
                            if 'eventId' in eventassist:
                                if event['teamId'] == eventassist['teamId']:
                                    if event['relatedEventId'] == eventassist['eventId']:
                                        shotAssistID = eventassist['eventId']
                                        shotChanceCreatorID = None
                                        shotChanceX1 = eventassist['x']
                                        shotChanceY1 = eventassist['y']
                                        for qualifiersb in eventassist['qualifiers']:
                                            if 'PassEndX' in qualifiersb['type']['displayName']:
                                                shotChanceX2 = qualifiersb['value']
                                            if 'PassEndY' in qualifiersb['type']['displayName']:
                                                shotChanceY2 = qualifiersb['value']
                                            if 'Cross' in qualifiersb['type']['displayName']:
                                                shotCrossYN = 1
                                            if 'Throughball' in qualifiersb['type']['displayName']:
                                                shotThroughballYN = 1
                                            if 'FreekickTaken' in qualifiersb['type']['displayName']:
                                                shotIndirectFKYN = 1
                    eventCount = 0
                    breakVal = 0
                    eventCountOwn = 0
                    shotSecondThroughballYN = 0
                    shotDribbleKeeperYN = 0
                    shotDribbleBeforeYN = 0
                    shotReboundYN = 0
                    shotErrorYN = 0
                    for iDNum in range(iD):
                        eventCount += 1
                        iDEvent = iD - (iDNum + 1)
                        if 'eventId' in gameData['events'][iDEvent]:
                            eventID = gameData['events'][iDEvent]['eventId']
                        else:
                            eventID = None
                        if gameData['events'][iDEvent]['teamId'] == shotTeamID:
                            eventCountOwn += 1
                            for qevent in gameData['events'][iDEvent]['qualifiers']:
                                if eventID != shotAssistID:
                                    if 'Throughball' in qevent['type']['displayName']:
                                        if shotAssistID != None:
                                            if eventID == ((shotAssistID) - 1):
                                                shotSecondThroughballYN = 1
                                if 'CornerTaken' in qevent['type']['displayName']:
                                    breakVal = 1
                                if 'FreekickTaken' in qevent['type']['displayName']:
                                    breakVal = 1
                                if 'ThrowIn' in qevent['type']['displayName']:
                                    breakVal = 1
                            if eventCount == 1:
                                if 'TakeOn' in gameData['events'][iDEvent]['type']['displayName']:
                                    if 'Successful' in gameData['events'][iDEvent]['outcomeType']['displayName']:
                                        for aevent in gameData['events'][iDEvent]['qualifiers']:
                                            if 'OppositeRelatedEvent' in aevent['type']['displayName']:
                                                dribbledID = aevent['value']
                                        for oevent in gameData['events']:
                                            if 'eventId' in oevent:
                                                if 'dribbledID' in locals():
                                                    if oevent['eventId'] == int(dribbledID) and oevent['teamId'] != shotTeamID:
                                                        if 'playerId' in oevent:
                                                            if oevent['playerId'] in gk_home_id:
                                                                shotDribbleKeeperYN = 1
                                                            elif oevent['playerId'] in gk_away_id:
                                                                shotDribbleKeeperYN = 1
                                                        else:
                                                            shotDribbleBeforeYN = 1
                                    if 'Unsuccessful' in gameData['events'][iDEvent]['outcomeType']['displayName']:
                                        breakVal = 1
                            if 'Tackle' in gameData['events'][iDEvent]['type']['displayName']:
                                breakVal = 1
                            if 'Interception' in gameData['events'][iDEvent]['type']['displayName']:
                                if 'Successful' in gameData['events'][iDEvent]['outcomeType']['displayName']:
                                    breakVal = 1
                            if 'Dispossessed' in gameData['events'][iDEvent]['type']['displayName']:
                                if 'Successful' in gameData['events'][iDEvent]['outcomeType']['displayName']:
                                    breakVal = 1
                            if 'OffsidePass' in gameData['events'][iDEvent]['type']['displayName']:
                                if 'Successful' in gameData['events'][iDEvent]['outcomeType']['displayName']:
                                    breakVal = 1
                            if 'Pass' in gameData['events'][iDEvent]['type']['displayName']:
                                if 'Unsuccessful' in gameData['events'][iDEvent]['outcomeType']['displayName']:
                                    breakVal = 1
                            if 'Foul' in gameData['events'][iDEvent]['type']['displayName']:
                                if 'Unsuccessful' in gameData['events'][iDEvent]['outcomeType']['displayName']:
                                    breakVal = 1
                            if 'isShot' in gameData['events'][iDEvent]:
                                breakVal = 1
                                if eventCountOwn == 1:
                                    if 10 not in gameData['events'][iDEvent]['satisfiedEventsTypes']:
                                        shotReboundYN = 1
                            if 'BallTouch' in gameData['events'][iDEvent]['type']['displayName']:
                                if 'Unsuccessful' in gameData['events'][iDEvent]['outcomeType']['displayName']:
                                    breakVal = 1
                        else:
                            if 'Error' in gameData['events'][iDEvent]['type']['displayName']:
                                shotErrorYN = 1
                            if 'Start' in gameData['events'][iDEvent]['type']['displayName']:
                                breakVal = 1
                        if breakVal == 1:
                            break
                eventDict = {'league': league, 'season':season,'homeTeam':homeTeam,'awayTeam':awayTeam,'date':date,'shotTeam':shotTeam,'shotMin':shotMin,
                                            'shotSec': shotSec, 'shotX':shotX,'shotY':shotY,'shotGoalYN':shotGoalYN,'shotState':shotState,'shotHeaderYN':shotHeaderYN,
                                            'shotBigChanceYN': shotBigChanceYN, 'shotFromCornerYN':shotFromCornerYN,'shotFastBreakYN':shotFastBreakYN,'shotPenaltyYN':shotPenaltyYN,
                                            'shotDirectFKYN': shotDirectFKYN, 'shotOwnGoalYN':shotOwnGoalYN,'shotChanceX1':shotChanceX1,'shotChanceY1':shotChanceY1,
                                            'shotChanceX2': shotChanceX2, 'shotChanceY2':shotChanceY2,'shotCrossYN':shotCrossYN,'shotThroughballYN':shotThroughballYN,
                                            'shotIndirectFKYN': shotIndirectFKYN, 'shotSecondThroughballYN': shotSecondThroughballYN, 'shotDribbleKeeperYN':shotDribbleKeeperYN, 
                                            'shotDribbleBeforeYN': shotDribbleBeforeYN, 'shotReboundYN': shotReboundYN, 'shotErrorYN':shotErrorYN, 'shotOnTarget':shotOnTarget,
                                            'shot6Yard': shot6Yard, 'shotPenaltyArea':shotPenaltyArea,'shotOutBox':shotOutBox}
                shotDatabase.append(eventDict)

        with open((OUTPUT_PATH + "shots_{}{}.csv".format(league_var, season_var)), 'a') as shotsfile:
            fieldnames = ['league', 'season', 'homeTeam','awayTeam','date','shotTeam','shotMin','shotSec','shotX','shotY','shotGoalYN','shotState','shotHeaderYN',
                                        'shotBigChanceYN', 'shotFromCornerYN', 'shotFastBreakYN','shotPenaltyYN','shotDirectFKYN','shotOwnGoalYN','shotChanceX1','shotChanceY1',
                                        'shotChanceX2', 'shotChanceY2', 'shotCrossYN','shotThroughballYN','shotIndirectFKYN','shotSecondThroughballYN','shotDribbleKeeperYN',
                                        'shotDribbleBeforeYN', 'shotReboundYN', 'shotErrorYN','shotOnTarget','shot6Yard','shotPenaltyArea','shotOutBox']
            writer = csv.DictWriter(shotsfile, fieldnames=fieldnames)
            for ia in shotDatabase:
                writer.writerow(ia)
            shotsfile.close()


for season in list_of_seasons:
    for league in list_of_leagues:
        getJSON(league, season)
        print('league:{}, season:{}'.format(league, season))
