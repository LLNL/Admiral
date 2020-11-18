/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 The Boeing Company
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

// This script configures two nodes on an 802.11b physical layer, with
// 802.11b NICs in adhoc mode, and by default, sends one packet of 1000
// (application) bytes to the other node.  The physical layer is configured
// to receive at a fixed RSS (regardless of the distance and transmit
// power); therefore, changing position of the nodes has no effect.
//
// There are a number of command-line options available to control
// the default behavior.  The list of available command-line options
// can be listed with the following command:
// ./waf --run "wifi-simple-adhoc --help"
//
// For instance, for this configuration, the physical layer will
// stop successfully receiving packets when rss drops below -97 dBm.
// To see this effect, try running:
//
// ./waf --run "wifi-simple-adhoc --rss=-97 --numPackets=20"
// ./waf --run "wifi-simple-adhoc --rss=-98 --numPackets=20"
// ./waf --run "wifi-simple-adhoc --rss=-99 --numPackets=20"
//
// Note that all ns-3 attributes (not just the ones exposed in the below
// script) can be changed at command line; see the documentation.
//
// This script can also be helpful to put the Wifi layer into verbose
// logging mode; this command will turn on all wifi logging:
//
// ./waf --run "wifi-simple-adhoc --verbose=1"
//
// When you are done, you will notice two pcap trace files in your directory.
// If you have tcpdump installed, you can try this:
//
// tcpdump -r wifi-simple-adhoc-0-0.pcap -nn -tt
//

#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/mobility-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/mobility-model.h"
#include "ns3/internet-stack-helper.h"

#include "ns3/ns3-ai-module.h"
#include "tcp-rl.h"
#include "tcp-rl-env.h"

using namespace ns3;
/*struct sTcpRlEnv
{
  uint32_t nodeId;
  uint32_t socketUid;
  uint8_t envType;
  int64_t simTime_us;
  uint32_t ssThresh;
  uint32_t cWnd;
  uint32_t segmentSize;
  uint32_t segmentsAcked;
  uint32_t bytesInFlight;
  //   int64_t rtt;
  //   int64_t minRtt;
  //   uint32_t calledFunc;
  //   uint32_t congState;
  //   uint32_t event;
  //   uint32_t ecnState;
} Packed;
struct TcpRlAct
{
  uint32_t new_ssThresh;
  uint32_t new_cWnd;
};
// typedef enum {
//   GET_SS_THRESH = 0,
//   INCREASE_WINDOW,
//   PKTS_ACKED,
//   CONGESTION_STATE_SET,
//   CWND_EVENT,
// } CalledFunc_t;
class TcpRlEnv : public Ns3AIRL<sTcpRlEnv, TcpRlAct>
{
public:

private:

}
*/
NS_LOG_COMPONENT_DEFINE ("WifiSimpleAdhoc");

TcpTimeStepEnv * my_env = new TcpTimeStepEnv(1234);

//Time myTimeStep{MilliSeconds(10)}

void ReceivePacket (Ptr<Socket> socket)
{
  static int packetsReceived = 0;

  while (socket->Recv ())
    {
      NS_LOG_UNCOND ("Received one packet! " << ++packetsReceived);
      auto env = my_env->EnvSetterCond();
      env->socketUid = 2;
      env->envType = 1;
      env->simTime_us = Simulator::Now().GetMicroSeconds();
      env->nodeId = 5;
      env->segmentSize = 200;
      NS_LOG_UNCOND ("Time: " << (uint64_t)(Simulator::Now().GetMilliSeconds()) << " env->ssThresh: " << env->ssThresh << " env->cWnd: "
                << env->cWnd << " env->segmentSize: " << env->segmentSize);
      my_env->SetCompleted();
  //    auto act = my_env->ActionGetterCond();
  //    NS_LOG_UNCOND ("Time: " << Simulator::Now().GetMilliSeconds() << " new_cWnd: " << act->new_cWnd << " new_ssThresh: " << act->new_ssThresh);
  //    my_env->GetCompleted();
    }
  NS_LOG_UNCOND("ReceivePacket: function exited.");
}

static void GenerateTraffic (Ptr<Socket> socket, uint32_t pktSize,
                             uint32_t pktCount, Time pktInterval )
{
  auto act = my_env->ActionGetterCond();
  static bool firstTime = true;
  static bool endSimulation = false;
  static int numPackets = 0;

  if(firstTime == true){
    firstTime = false;
    socket->Send (Create<Packet> (pktSize));
    ++numPackets;
  }

  if(act->new_ssThresh == 1){
    NS_LOG_UNCOND ("Time: " << Simulator::Now().GetMilliSeconds() << " new_cWnd: " << act->new_cWnd << " new_ssThresh: " << act->new_ssThresh);
    act->new_ssThresh = 0;
    ++numPackets;
    NS_LOG_UNCOND("GenerateTraffic: Sending new packet. PktCount: " << numPackets);
    socket->Send (Create<Packet> (pktSize));
  } else if (act->new_ssThresh == 1234) {
    NS_LOG_UNCOND("GenerateTraffic: Closing socket.");
    socket->Close ();
    my_env->SetFinish();
    endSimulation = true;
    NS_LOG_UNCOND("GenerateTraffic: Socket closed, environment finished.");
  } else {
    NS_LOG_UNCOND("ERROR, incorrect state.");
  }
  if(endSimulation == false){
    Simulator::Schedule (pktInterval, &GenerateTraffic,
                         socket, pktSize,pktCount, pktInterval);
  }
}

int main (int argc, char *argv[])
{
  std::string phyMode ("DsssRate1Mbps");
  double rss = -80;  // -dBm
  uint32_t packetSize = 1000; // bytes
  uint32_t numPackets = 1;
  double interval = 1.0; // seconds
  bool verbose = false;
  //double duration = 20.0;

  // Set the simulation start and stop time
  //double start_time = 0.1;
  //double stop_time = start_time + duration;


  CommandLine cmd;
  cmd.AddValue ("phyMode", "Wifi Phy mode", phyMode);
  cmd.AddValue ("rss", "received signal strength", rss);
  cmd.AddValue ("packetSize", "size of application packet sent", packetSize);
  cmd.AddValue ("numPackets", "number of packets generated", numPackets);
  cmd.AddValue ("interval", "interval (seconds) between packets", interval);
  cmd.AddValue ("verbose", "turn on all WifiNetDevice log components", verbose);
  cmd.Parse (argc, argv);
  // Convert to time object
  Time interPacketInterval = Seconds (interval);

  // Fix non-unicast data rate to be the same as that of unicast
  Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode",
                      StringValue (phyMode));

  NodeContainer c;
  c.Create (2);

  // The below set of helpers will help us to put together the wifi NICs we want
  WifiHelper wifi;
  if (verbose)
    {
      wifi.EnableLogComponents ();  // Turn on all Wifi logging
    }
  wifi.SetStandard (WIFI_PHY_STANDARD_80211b);
//  wifi.setNoiseFigure(1.0);

  YansWifiPhyHelper wifiPhy =  YansWifiPhyHelper::Default ();
  // This is one parameter that matters when using FixedRssLossModel
  // set it to zero; otherwise, gain will be added
  wifiPhy.Set ("RxGain", DoubleValue (0) );
  //wifiPhy.Set ("RxNoiseFigure", DoubleValue (106.551));

  // ns-3 supports RadioTap and Prism tracing extensions for 802.11b
  wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);

  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  // The below FixedRssLossModel will cause the rss to be fixed regardless
  // of the distance between the two stations, and the transmit power
  wifiChannel.AddPropagationLoss ("ns3::FixedRssLossModel","Rss",DoubleValue (rss));
  wifiPhy.SetChannel (wifiChannel.Create ());
  //wifiPhy.SetErrorRateModel ("ns3::RateErrorModel");

  // Add a mac and disable rate control
  WifiMacHelper wifiMac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode",StringValue (phyMode),
                                "ControlMode",StringValue (phyMode));
  // Set it to adhoc mode
  wifiMac.SetType ("ns3::AdhocWifiMac");
  NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, c);

  // Note that with FixedRssLossModel, the positions below are not
  // used for received signal strength.
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (c);

  InternetStackHelper internet;
  internet.Install (c);

  Ipv4AddressHelper ipv4;
  NS_LOG_UNCOND ("Assign IP Addresses.");
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i = ipv4.Assign (devices);

  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  Ptr<Socket> recvSink = Socket::CreateSocket (c.Get (0), tid);
  InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 80);
  recvSink->Bind (local);
  recvSink->SetRecvCallback (MakeCallback (&ReceivePacket));

  Ptr<Socket> source = Socket::CreateSocket (c.Get (1), tid);
  InetSocketAddress remote = InetSocketAddress (Ipv4Address ("255.255.255.255"), 80);
  source->SetAllowBroadcast (true);
  source->Connect (remote);

  // Tracing
  wifiPhy.EnablePcap ("wifi-simple-adhoc", devices);

  // Output what we are doing
  NS_LOG_UNCOND ("Testing " << numPackets  << " packets sent with receiver rss " << rss );
  auto env = Create<TcpTimeStepEnv> (1234);
  NS_LOG_UNCOND ("CreateEnv: " << (env == 0));
  Simulator::ScheduleWithContext (source->GetNode ()->GetId (),
                                  Seconds (1.0), &GenerateTraffic,
                                  source, packetSize, numPackets, interPacketInterval);
  //Simulator::Stop (Seconds (stop_time));
  Simulator::Run ();
  Simulator::Destroy ();

  NS_LOG_UNCOND("Simulation finished.");

  return 0;
}