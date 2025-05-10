# pyclient_telemetry.py

import sys
import argparse
import socket
import msgParser
import csv
import os
import time

# python pyclient_telemetry.py --track yourTrackName --maxEpisodes 2 --maxSteps 5000
# python pyclient_telemetry.py --track oval --maxEpisodes 2 --maxSteps 5000
# python pyclient_telemetry.py --track oval-p406 --maxEpisodes 3 --maxSteps 10000

# python pyclient_telemetry.py --track oval-pw-evoviwrc --maxEpisodes 3 --maxSteps 10000



# import your driver here (it should expose .drive() and .init(), and control attributes)
# import driver as driver_module  
import driver_model as driver_module  

def main():
    # 1) Command‑line args
    parser = argparse.ArgumentParser(
        description='TORCS client with buffered telemetry logging for RL.'
    )
    parser.add_argument('--host',       default='localhost', help='TORCS server IP')
    parser.add_argument('--port', type=int, default=3001,       help='TORCS server port')
    parser.add_argument('--id',         default='SCR',         help='Bot ID')
    parser.add_argument('--maxEpisodes',type=int, default=1,    help='Number of episodes')
    parser.add_argument('--maxSteps',   type=int, default=0,    help='Max steps per episode (0 = unlimited)')
    parser.add_argument('--track',      default='unknown',     help='Track name for logging')
    parser.add_argument('--stage',      type=int, default=3,    help='Stage: 0=WarmUp,1=Qual,2=Race,3=Unknown')
    args = parser.parse_args()

    # 2) Socket setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)

    # 3) Prepare driver & parser
    driver = driver_module.Driver(args.stage, manual_mode=True)
    parser_msg = msgParser.MsgParser()

    # 4) CSV setup
    csv_filename = f'telemetry_{args.track}.csv'
    fieldnames = [
        'timestamp', 'trackName',
        # scalar sensors
        'angle','curLapTime','damage','distFromStart','distRaced',
        'fuel','gear','lastLapTime','racePos','rpm',
        'speedX','speedY','speedZ','trackPos','z',
        # vector sensors
        *[f'track_{i}'    for i in range(19)],
        *[f'opponent_{i}' for i in range(36)],
        *[f'focus_{i}'    for i in range(5)],
        # actions
        'accel','brake','steer','gear_cmd','clutch','meta'
    ]
    buffered_rows = []

    shutdown = False
    episode  = 0

    while not shutdown:
        # ==== Handshake ====
        while True:
            init_msg = args.id + driver.init()
            sock.sendto(init_msg.encode(), (args.host, args.port))
            try:
                data, _ = sock.recvfrom(1024)
                text = data.decode()
            except socket.error:
                continue
            if '***identified***' in text:
                print(f"[Episode {episode+1}] Identified by server.")
                break

        step = 0
        # ==== Main loop ====
        while True:
            try:
                data, _ = sock.recvfrom(4096)
                text = data.decode()
            except socket.error:
                continue

            # shutdown / restart signals
            if '***shutdown***' in text:
                driver.onShutDown()
                shutdown = True
                break
            if '***restart***' in text:
                driver.onRestart()
                break

            # parse sensors & compute action
            sensors = parser_msg.parse(text)
            action  = driver.drive(text)
            ctrl    = driver.control

            # build one row
            row = {
                'timestamp':    time.time(),
                'trackName':    args.track,
                'angle':        sensors.get('angle',       [''])[0],
                'curLapTime':   sensors.get('curLapTime',  [''])[0],
                'damage':       sensors.get('damage',      [''])[0],
                'distFromStart':sensors.get('distFromStart',[''])[0],
                'distRaced':    sensors.get('distRaced',   [''])[0],
                'fuel':         sensors.get('fuel',        [''])[0],
                'gear':         sensors.get('gear',        [''])[0],
                'lastLapTime':  sensors.get('lastLapTime', [''])[0],
                'racePos':      sensors.get('racePos',     [''])[0],
                'rpm':          sensors.get('rpm',         [''])[0],
                'speedX':       sensors.get('speedX',      [''])[0],
                'speedY':       sensors.get('speedY',      [''])[0],
                'speedZ':       sensors.get('speedZ',      [''])[0],
                'trackPos':     sensors.get('trackPos',    [''])[0],
                'z':            sensors.get('z',           [''])[0],
                # actions
                'accel':        ctrl.getAccel(),
                'brake':        ctrl.getBrake(),
                'steer':        ctrl.getSteer(),
                'gear_cmd':     ctrl.getGear(),
                'clutch':       ctrl.getClutch(),
                'meta':         ctrl.getMeta()
            }
            # expand vector sensors
            for i, v in enumerate(sensors.get('track',    [0]*19)):
                row[f'track_{i}']    = v
            for i, v in enumerate(sensors.get('opponents',[0]*36)):
                row[f'opponent_{i}'] = v
            for i, v in enumerate(sensors.get('focus',    [0]*5)):
                row[f'focus_{i}']    = v

            buffered_rows.append(row)

            # send action or meta
            step += 1
            if args.maxSteps>0 and step >= args.maxSteps:
                sock.sendto('(meta 1)'.encode(), (args.host, args.port))
            else:
                sock.sendto(action.encode(), (args.host, args.port))

        episode += 1
        if episode >= args.maxEpisodes:
            shutdown = True

    sock.close()

    # ==== Save prompt ====
    if not buffered_rows:
        print("No telemetry collected.")
        return

    print(f"\nCollected {len(buffered_rows)} rows.")
    save = input(f"Save to '{csv_filename}'? (y/N): ").strip().lower()
    if save == 'y':
        write_header = not os.path.exists(csv_filename)
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(buffered_rows)
        print(f"Data saved to {csv_filename}")
    else:
        print("Telemetry discarded.")

if __name__ == '__main__':
    main()
