import { SignFrameCNN } from './model.js';

const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), overlayCtx = overlay.getContext('2d');
const frameCanvas = document.createElement('canvas');
const frameCtx = frameCanvas.getContext('2d', {willReadFrequently:true});
const crop = $('cropPreview'), cropCtx = crop.getContext('2d');
const msg = $('message');
let landmarker=null, cnn=null, stream=null, running=false, busy=false;
let stable=null, history=[], nextFrame=0;
const CONNECTIONS=[
 [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
 [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
 [13,17],[17,18],[18,19],[19,20],[0,17]
];

function status(message){ $('status').textContent=message; }
function resetPrediction(){
 history=[]; stable=null;
 $('letter').textContent='–'; $('confidence').textContent='0%';
 $('confidenceBar').style.width='0%'; $('stableLetter').textContent='Waiting';
}
function updatePrediction(letter,confidence){
 $('letter').textContent=letter;
 $('confidence').textContent=Math.round(confidence*100)+'%';
 $('confidenceBar').style.width=(100*confidence).toFixed(1)+'%';
 history.push(confidence>=0.65?letter:null);
 if(history.length>12) history.shift();
 const counts=new Map();
 for(const value of history) if(value) counts.set(value,(counts.get(value)||0)+1);
 stable=null;
 for(const [letter,count] of counts) if(count>=9){stable=letter;break;}
 $('stableLetter').textContent=stable||'Waiting';
}
function getBox(points,width,height){
 const xs=points.map(p=>p.x*width),ys=points.map(p=>p.y*height);
 const cx=(Math.min(...xs)+Math.max(...xs))/2,cy=(Math.min(...ys)+Math.max(...ys))/2;
 const span=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys));
 const side=Math.max(80,Math.round(span*1.25));
 return {x1:Math.round(cx-side/2),y1:Math.round(cy-side/2),side};
}
function renderHand(points,box,width,height){
 overlayCtx.clearRect(0,0,width,height);
 overlayCtx.strokeStyle='#8df7ac';
 overlayCtx.lineWidth=Math.max(2,width/350);
 for(const [a,b] of CONNECTIONS){
  overlayCtx.beginPath();
  overlayCtx.moveTo(points[a].x*width,points[a].y*height);
  overlayCtx.lineTo(points[b].x*width,points[b].y*height);
  overlayCtx.stroke();
 }
 overlayCtx.fillStyle='#f19e91';
 for(const point of points){
  overlayCtx.beginPath();
  overlayCtx.arc(point.x*width,point.y*height,3.2,0,Math.PI*2);
  overlayCtx.fill();
 }
 overlayCtx.lineWidth=3;
 overlayCtx.strokeRect(box.x1,box.y1,box.side,box.side);
}

/** Bilinear 28x28 grayscale crop. The 0.299/0.587/0.114 weights mimic OpenCV BGR-to-gray. */
function preprocessImage(image,width,height,box){
 const pixels=image.data, out=new Float32Array(784);
 const sample=(fx,fy,channel)=>{
  const x=Math.max(0,Math.min(width-1,fx));
  const y=Math.max(0,Math.min(height-1,fy));
  const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(width-1,x0+1),y1=Math.min(height-1,y0+1);
  const dx=x-x0,dy=y-y0;
  const a=pixels[(y0*width+x0)*4+channel];
  const b=pixels[(y0*width+x1)*4+channel];
  const c=pixels[(y1*width+x0)*4+channel];
  const d=pixels[(y1*width+x1)*4+channel];
  return (a*(1-dx)+b*dx)*(1-dy)+(c*(1-dx)+d*dx)*dy;
 };
 const preview=cropCtx.createImageData(28,28);
 for(let y=0;y<28;y++)for(let x=0;x<28;x++){
  const sx=box.x1+(x+0.5)*box.side/28-0.5;
  const sy=box.y1+(y+0.5)*box.side/28-0.5;
  const gray=0.299*sample(sx,sy,0)+0.587*sample(sx,sy,1)+0.114*sample(sx,sy,2);
  const index=y*28+x;
  out[index]=gray/255;
  preview.data[4*index]=preview.data[4*index+1]=preview.data[4*index+2]=Math.round(gray);
  preview.data[4*index+3]=255;
 }
 const small=document.createElement('canvas');small.width=28;small.height=28;
 small.getContext('2d').putImageData(preview,0,0);
 cropCtx.clearRect(0,0,crop.width,crop.height);
 cropCtx.imageSmoothingEnabled=false;
 cropCtx.drawImage(small,0,0,crop.width,crop.height);
 return out;
}

async function start(){
 if(running||busy)return;
 busy=true;$('cameraBtn').disabled=true;
 status('Loading CNN weights and MediaPipe hand detector…');
 try{
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)
   throw Error('Camera access requires HTTPS or http://localhost');
  if(!cnn)cnn=await SignFrameCNN.load('./model/');
  if(!landmarker){
   const vision=await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs');
   const fileset=await vision.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm');
   landmarker=await vision.HandLandmarker.createFromOptions(fileset,{
    baseOptions:{
     modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
     delegate:'CPU'
    },
    runningMode:'VIDEO',numHands:1,
    minHandDetectionConfidence:0.5,minHandPresenceConfidence:0.5,minTrackingConfidence:0.5
   });
  }
  stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:960},height:{ideal:540}},audio:false});
  video.srcObject=stream;await video.play();
  const w=video.videoWidth,h=video.videoHeight;
  frameCanvas.width=w;frameCanvas.height=h;
  overlay.width=w;overlay.height=h;
  running=true;nextFrame=0;
  $('cameraBlank').classList.add('hidden');
  $('liveState').textContent='LIVE';$('liveState').classList.add('on');
  $('stopBtn').disabled=false;
  status('Camera ready. Show one static ASL letter, and hold it steady.');
  requestAnimationFrame(loop);
 }catch(error){
  stop();status('Unable to start: '+(error?.message||error));
 }finally{
  busy=false;if(!running)$('cameraBtn').disabled=false;
 }
}
function loop(now){
 if(!running)return;
 requestAnimationFrame(loop);
 if(now<nextFrame)return;
 nextFrame=now+270;
 try{
  const w=frameCanvas.width,h=frameCanvas.height;
  frameCtx.save();frameCtx.translate(w,0);frameCtx.scale(-1,1);
  frameCtx.drawImage(video,0,0,w,h);frameCtx.restore();
  const detection=landmarker.detectForVideo(frameCanvas,now);
  const hand=detection.landmarks?.[0];
  if(!hand){resetPrediction();overlayCtx.clearRect(0,0,w,h);return;}
  const box=getBox(hand,w,h);
  renderHand(hand,box,w,h);
  const grayscale=preprocessImage(frameCtx.getImageData(0,0,w,h),w,h,box);
  const prediction=cnn.predict(grayscale);
  updatePrediction(prediction.letter,prediction.confidence);
 }catch(error){
  stop();status('Recognition error: '+(error?.message||error));
 }
}
function stop(){
 running=false;
 if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
 video.srcObject=null;
 $('cameraBtn').disabled=false;$('stopBtn').disabled=true;
 $('cameraBlank').classList.remove('hidden');
 $('liveState').textContent='OFFLINE';$('liveState').classList.remove('on');
 overlayCtx.clearRect(0,0,overlay.width,overlay.height);
 resetPrediction();
}
function addLetter(){
 if(stable){
  msg.value+=stable;
  status('Added: '+stable);
  history=[];stable=null;$('stableLetter').textContent='Waiting';
 }else status('Hold a sign until a stable prediction appears.');
}
function speak(){
 if(!('speechSynthesis' in window)){status('Speech not supported in this browser');return;}
 if(!msg.value.trim()){status('Your message is empty');return;}
 window.speechSynthesis.cancel();
 const utterance=new SpeechSynthesisUtterance(msg.value.trim());
 utterance.lang='en-US';window.speechSynthesis.speak(utterance);
 status('Speaking message…');
}
$('cameraBtn').addEventListener('click',start);
$('stopBtn').addEventListener('click',()=>{stop();status('Camera stopped');});
$('addBtn').addEventListener('click',addLetter);
$('spaceBtn').addEventListener('click',()=>{msg.value+=' ';msg.focus();});
$('backspaceBtn').addEventListener('click',()=>{msg.value=msg.value.slice(0,-1);msg.focus();});
$('speakBtn').addEventListener('click',speak);
$('clearBtn').addEventListener('click',()=>{msg.value='';resetPrediction();status('Message cleared');});
document.addEventListener('keydown',event=>{
 if(document.activeElement===msg||event.ctrlKey||event.altKey||event.metaKey)return;
 if(event.key==='Enter'){event.preventDefault();addLetter();}
 else if(event.key===' '){event.preventDefault();msg.value+=' ';}
 else if(event.key==='Backspace'){event.preventDefault();msg.value=msg.value.slice(0,-1);}
 else if(event.key.toLowerCase()==='v')speak();
 else if(event.key.toLowerCase()==='c'){msg.value='';resetPrediction();}
});
window.addEventListener('beforeunload',stop);
