import { Component, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { FilesService } from './services/files.service';
import { Title } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  constructor(private titleService:Title, public sanitizer: DomSanitizer, public fileService: FilesService){}
  title = 'app';
  welcomePage = "./assets/files/testhtml2.htm";
  //root: string = "./dist/portfolio";
  filepath: string = this.welcomePage;
  //currentpath = "/assets/files";
  openMenu = true;
  ngOnInit(): void{
    console.log("init...");
    this.titleService.setTitle("Xuan Yang");
    /*
    this.fileService.getFiles(this.appendtoRoot(this.currentpath)).subscribe(res => {
      console.log(res);
      this.files = JSON.parse(res);
      console.log(this.files.name);
    });
    */
  }
  /*
  getDir(name){
    this.currentpath = this.currentpath.concat("/").concat(name);
    
    this.fileService.getFiles(this.appendtoRoot(this.currentpath)).subscribe(res => {
      this.files = JSON.parse(res);
    });
    
  }
  getFile(name){
    this.filepath = this.currentpath.concat("/").concat(name);
    console.log(this.filepath);
  }
  */
  return(){
    this.filepath = this.welcomePage;
    /*
    if(this.currentpath.length < 14){
      this.filepath = this.welcomePage;
      return;
    }
    var temp = this.currentpath.lastIndexOf("/");
    this.currentpath = this.currentpath.substring(0, temp);
    console.log(this.currentpath);
    
    this.fileService.getFiles(this.appendtoRoot(this.currentpath)).subscribe(res => {
      console.log(res);
      this.files = JSON.parse(res);
    });
    */
  }

/*
  appendtoRoot(path){
    return this.root.concat(path);
  }

*/

/*
  css help functions

*/
  toggleMenu(){
    console.log("hishis");
    $(".arrow--l-r").toggleClass("left right");
    
  }
  toggleSubMenu(id){
    var ele = document.getElementById(id);
    console.log(ele.style.maxHeight);
    if(ele.style.maxHeight != '200px'){
      ele.style.maxHeight = '200px';
    }else{
      ele.style.maxHeight = '0px';
    }
  }
  
}
